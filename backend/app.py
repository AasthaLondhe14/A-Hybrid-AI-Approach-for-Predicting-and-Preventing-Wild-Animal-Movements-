from flask import Flask, jsonify, send_from_directory, request
from ultralytics import YOLO
import cv2
import os
import tempfile
import subprocess
import wave
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import csv
import sounddevice as sd
from flask_cors import CORS

# Ensure TFHub cache is a valid directory inside this backend folder
TFHUB_CACHE_DIR = os.path.join(os.path.dirname(__file__), "tfhub_cache")
os.makedirs(TFHUB_CACHE_DIR, exist_ok=True)
os.environ["TFHUB_CACHE_DIR"] = TFHUB_CACHE_DIR

app = Flask(__name__, static_folder="static")
CORS(app)

# Load YOLOv8 model
model = YOLO(os.path.join("yolov8", "best.pt"))

# IP camera stream (update if your camera uses a different path)
ip_camera_url = "http://192.0.0.4:8080/video"

# IP camera audio stream (set to your camera's audio/rtsp stream if different)
ip_camera_audio_url = os.getenv("IP_CAMERA_AUDIO_URL", "")

# Load YAMNet for audio classification (pretrained on AudioSet)
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")

def class_names_from_csv(class_map_csv_text):
    class_names = []
    with tf.io.gfile.GFile(class_map_csv_text) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            class_names.append(row["display_name"])
    return class_names

yamnet_class_map_path = yamnet_model.class_map_path().numpy()
yamnet_class_names = class_names_from_csv(yamnet_class_map_path)

TARGET_KEYWORDS = {
    "lion": ["roaring cats", "roar", "growl"],
    "tiger": ["roaring cats", "roar", "growl"],
    "cheetah": ["roaring cats", "cat", "growl"],
    "cat": ["cat", "meow", "purr"],
    "dog": ["dog", "bark", "growl"],
    "human": ["human voice", "speech", "scream", "shout"],
    "cow": ["cattle", "cow", "moo"],
    "deer": ["deer"],
}

def build_target_indices():
    indices = {}
    for target, keywords in TARGET_KEYWORDS.items():
        idxs = []
        for i, name in enumerate(yamnet_class_names):
            lname = name.lower()
            if any(k in lname for k in keywords):
                idxs.append(i)
        indices[target] = idxs
    return indices

target_indices = build_target_indices()

def capture_audio_wav(url, seconds=3):
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
        "16000",
        "-f",
        "wav",
        tmp.name,
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=12)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode("utf-8", errors="ignore")[-400:])
    return tmp.name

def capture_audio_mic(seconds=3, sample_rate=16000):
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
    scores, embeddings, spectrogram = yamnet_model(waveform)
    scores_np = scores.numpy()
    frame_max = np.max(scores_np, axis=0)
    frame_mean = np.mean(scores_np, axis=0)
    combined = (0.7 * frame_max) + (0.3 * frame_mean)
    target_scores = {}
    for target, idxs in target_indices.items():
        if idxs:
            target_scores[target] = float(np.max(combined[idxs]))
        else:
            target_scores[target] = 0.0
    return target_scores

# Local fallback video (kept for testing if IP cam is down)
video_filename = "LionM.mp4"
video_path = os.path.join("static", video_filename)

@app.route("/detect", methods=["GET"])
def detect_from_video():
    # Read directly from the IP camera stream
    cap = cv2.VideoCapture(ip_camera_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    detected_classes = set()
    class_counts = {}

    if not cap.isOpened():
        return jsonify({
            "status": "error",
            "message": "Could not open IP camera stream. Check the URL: " + ip_camera_url
        }), 500

    frame_count = 0
    while frame_count < 5:  # Process only 5 frames for speed
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

    return jsonify({
        "status": "success",
        "detected": list(detected_classes),
        "counts": class_counts
    })

@app.route("/audio_detect", methods=["GET"])
def detect_from_audio():
    seconds = int(request.args.get("seconds", 6))
    threshold = float(request.args.get("threshold", 0.12))
    try:
        # Use laptop mic if no IP camera audio URL is set
        if not ip_camera_audio_url:
            wav_path = capture_audio_mic(seconds=seconds)
        else:
            wav_path = capture_audio_wav(ip_camera_audio_url, seconds=seconds)
        sample_rate, waveform = load_waveform(wav_path)
        os.unlink(wav_path)
        if sample_rate != 16000:
            return jsonify({"status": "error", "message": "Audio sample rate is not 16kHz"}), 500
        scores = infer_audio(waveform)
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
    app.run(host="0.0.0.0", port=5000, debug=True)
