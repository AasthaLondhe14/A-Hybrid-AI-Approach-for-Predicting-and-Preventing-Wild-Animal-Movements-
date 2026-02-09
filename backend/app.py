from flask import Flask, jsonify, send_from_directory, request
from ultralytics import YOLO
import cv2
import os
import tempfile
import subprocess
import time
import wave
import numpy as np
import sounddevice as sd
import librosa
import urllib.request
import json
import torch
import torch.nn as nn
from flask_cors import CORS

app = Flask(__name__, static_folder="static")
CORS(app)

# Load YOLOv8 model
model = YOLO(os.path.join("yolov8", "best.pt"))

# IP camera stream (update if your camera uses a different path)
ip_camera_url = "http://100.104.143.1:8080/video"

# IP camera audio stream (set to your camera's audio/rtsp stream if different)
ip_camera_audio_url = os.getenv("IP_CAMERA_AUDIO_URL", "")

# PANNs assets
PANNS_DATA_DIR = os.path.join(os.path.expanduser("~"), "panns_data")
PANNS_LABELS_PATH = os.path.join(PANNS_DATA_DIR, "class_labels_indices.csv")
# Use a separate filename to avoid conflicts with partially locked downloads
PANNS_CHECKPOINT_PATH = os.path.join(PANNS_DATA_DIR, "Cnn14_mAP=0.431.pth.bin")
PANNS_LABELS_URL = "https://raw.githubusercontent.com/qiuqiangkong/audioset_tagging_cnn/master/metadata/class_labels_indices.csv"
PANNS_CHECKPOINT_URL = "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1"
PANNS_MIN_BYTES = 300_000_000  # Cnn14 checkpoint is large; re-download if smaller than 300MB

def download_file(url, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    urllib.request.urlretrieve(url, path)

def get_remote_size(url):
    try:
        with urllib.request.urlopen(url) as response:
            length = response.headers.get("Content-Length")
            return int(length) if length else None
    except Exception:
        return None

def ensure_panns_assets():
    os.makedirs(PANNS_DATA_DIR, exist_ok=True)
    if not os.path.exists(PANNS_LABELS_PATH):
        download_file(PANNS_LABELS_URL, PANNS_LABELS_PATH)
    expected_size = get_remote_size(PANNS_CHECKPOINT_URL)
    if os.path.exists(PANNS_CHECKPOINT_PATH):
        local_size = os.path.getsize(PANNS_CHECKPOINT_PATH)
        min_size = expected_size if expected_size else PANNS_MIN_BYTES
        if local_size < min_size:
            try:
                os.remove(PANNS_CHECKPOINT_PATH)
            except OSError:
                pass
    if not os.path.exists(PANNS_CHECKPOINT_PATH):
        download_file(PANNS_CHECKPOINT_URL, PANNS_CHECKPOINT_PATH)

ensure_panns_assets()

# Load PANNs audio tagging model (AudioSet pretrained)
from panns_inference import AudioTagging, labels as panns_labels
audio_tagger = AudioTagging(checkpoint_path=PANNS_CHECKPOINT_PATH, device="cpu")

MODEL_PATH = r"E:\\datasets\\audio_datasets\\models\\audio_classifier.pth"
LABELS_PATH = r"E:\\datasets\\audio_datasets\\models\\labels.json"

class MLP(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.net(x)

USE_FINETUNED_AUDIO = False
audio_classifier = None
audio_labels = None

def load_audio_classifier():
    global audio_classifier, audio_labels
    if not (os.path.exists(MODEL_PATH) and os.path.exists(LABELS_PATH)):
        return
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        audio_labels = json.load(f)
    # Infer embedding size from a dummy 10s input
    dummy = np.zeros((1, 32000 * 10), dtype=np.float32)
    _, embedding = audio_tagger.inference(dummy)
    emb = embedding[0]
    if emb.ndim > 1:
        emb = emb.mean(axis=0)
    audio_classifier = MLP(emb.shape[0], len(audio_labels))
    audio_classifier.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    audio_classifier.eval()

load_audio_classifier()

TARGET_KEYWORDS = {
    "lion": ["roar", "roaring", "growl", "big cat", "animal"],
    "tiger": ["roar", "roaring", "growl", "big cat", "animal"],
    "cheetah": ["cat", "growl", "animal"],
    "cat": ["cat", "meow", "purr"],
    "dog": ["dog", "bark", "growl"],
    "human": ["speech", "human", "scream", "shout"],
    "cow": ["cattle", "cow", "moo"],
    "deer": ["deer", "animal"],
}

def build_target_indices():
    indices = {}
    for target, keywords in TARGET_KEYWORDS.items():
        idxs = []
        for i, name in enumerate(panns_labels):
            lname = name.lower()
            if any(k in lname for k in keywords):
                idxs.append(i)
        indices[target] = idxs
    return indices

target_indices = build_target_indices()

def capture_audio_wav(url, seconds=6, sample_rate=32000):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp.close()
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        url,
        "-t",
        str(seconds),
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "wav",
        tmp.name,
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=12)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode("utf-8", errors="ignore")[-400:])
    return tmp.name

def capture_audio_mic(seconds=6, sample_rate=32000):
    audio = sd.rec(int(seconds * sample_rate), samplerate=sample_rate, channels=1, dtype="int16")
    sd.wait()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp.close()
    with wave.open(tmp.name, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())
    return tmp.name

def load_waveform(wav_path):
    with wave.open(wav_path, "rb") as wf:
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        audio = wf.readframes(n_frames)
        waveform = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
    return sample_rate, waveform

def infer_audio(waveform):
    # If fine-tuned classifier is available, use it
    if USE_FINETUNED_AUDIO and audio_classifier is not None and audio_labels is not None:
        if waveform.ndim == 1:
            waveform = waveform[None, :]
        _, embedding = audio_tagger.inference(waveform)
        emb = embedding[0]
        if emb.ndim > 1:
            emb = emb.mean(axis=0)
        with torch.no_grad():
            logits = audio_classifier(torch.from_numpy(emb).unsqueeze(0))
            probs = torch.softmax(logits, dim=1).squeeze(0).numpy()
        return {audio_labels[str(i)]: float(probs[i]) for i in range(len(probs))}

    # Fallback to generic AudioSet scores
    if waveform.ndim == 1:
        waveform = waveform[None, :]
    (clipwise_output, embedding) = audio_tagger.inference(waveform)
    scores_np = clipwise_output[0]
    target_scores = {}
    for target, idxs in target_indices.items():
        if idxs:
            target_scores[target] = float(np.max(scores_np[idxs]))
        else:
            target_scores[target] = 0.0
    return target_scores

# Local fallback video (kept for testing if IP cam is down)
video_filename = "LionM.mp4"
video_path = os.path.join("static", video_filename)

@app.route("/detect", methods=["GET"])
def detect_from_video():
    # Read directly from the IP camera stream with timeouts
    cap = cv2.VideoCapture()
    try:
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 3000)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 3000)
    except Exception:
        pass
    cap.open(ip_camera_url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(ip_camera_url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    detected_classes = set()
    class_counts = {}

    if not cap.isOpened():
        return jsonify({
            "status": "error",
            "message": "Could not open IP camera stream. Check the URL: " + ip_camera_url
        }), 500

    frame_count = 0
    max_wait_seconds = 6
    start_time = time.time()
    while frame_count < 5:  # Process only 5 frames for speed
        if time.time() - start_time > max_wait_seconds:
            break
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]
        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            detected_classes.add(label)
            class_counts[label] = class_counts.get(label, 0) + 1

        frame_count += 1

    cap.release()
    if frame_count == 0:
        return jsonify({
            "status": "error",
            "message": "No frames received from IP camera stream: " + ip_camera_url
        }), 500

    video_scores = {label: count / frame_count for label, count in class_counts.items()}

    return jsonify({
        "status": "success",
        "detected": list(detected_classes),
        "counts": class_counts,
        "scores": video_scores,
        "frames": frame_count
    })

@app.route("/audio_detect", methods=["GET"])
def detect_from_audio():
    seconds = int(request.args.get("seconds", 8))
    threshold = float(request.args.get("threshold", 0.08))
    try:
        # Use laptop mic if no IP camera audio URL is set
        if not ip_camera_audio_url:
            wav_path = capture_audio_mic(seconds=seconds, sample_rate=32000)
        else:
            wav_path = capture_audio_wav(ip_camera_audio_url, seconds=seconds, sample_rate=32000)
        sample_rate, waveform = load_waveform(wav_path)
        os.unlink(wav_path)
        if sample_rate != 32000:
            waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=32000)
        # Pad/crop to 10 seconds for PANNs
        target_len = 32000 * 10
        if len(waveform) < target_len:
            waveform = np.pad(waveform, (0, target_len - len(waveform)))
        elif len(waveform) > target_len:
            waveform = waveform[:target_len]
        scores = infer_audio(waveform)
        # Disable human completely from audio results
        if "human" in scores:
            scores.pop("human", None)

        detected = [k for k, v in scores.items() if v >= threshold]
        detected_sorted = sorted(detected, key=lambda k: scores[k], reverse=True)
        return jsonify({
            "status": "success",
            "detected": detected_sorted,
            "scores": scores
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# Route to serve local video file (optional fallback)
@app.route("/video")
def serve_video():
    return send_from_directory("static", video_filename)

if __name__ == "__main__":
    # Disable reloader to avoid Windows watchdog/socket issues with large ML deps
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
