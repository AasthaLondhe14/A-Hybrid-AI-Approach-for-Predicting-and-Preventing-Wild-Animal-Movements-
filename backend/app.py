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
import pickle
import torch
import torch.nn as nn
import tensorflow as tf
import tensorflow_hub as hub
from flask_cors import CORS
from database import store_detection, get_detection_history
from email_service import send_danger_alert_email

app = Flask(__name__, static_folder="static")
CORS(app)
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
YAMNET_CACHE_DIR = os.path.join(BACKEND_DIR, "tfhub_cache_yamnet")
os.makedirs(YAMNET_CACHE_DIR, exist_ok=True)
os.environ["TFHUB_CACHE_DIR"] = YAMNET_CACHE_DIR

# Load YOLOv8 model
model = YOLO(os.path.join("yolov8", "bestR.pt"))

# IP camera stream (update if your camera uses a different path)
ip_camera_url = "http://100.107.105.168:8080/video"

# IP camera audio stream (set to your camera's audio/rtsp stream if different)
ip_camera_audio_url = "http://100.107.105.168:8080/audio.wav"

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

MODEL_PATH = r"E:\\models\\audio_classifier_balanced.pth"
LABELS_PATH = r"E:\\models\\labels_balanced.json"
YAMNET_SVM_MODEL_PATH = r"E:\\models\\YAMNet_SVM_optimized_model.pkl"
YAMNET_SVM_LABEL_ENCODER_PATH = r"E:\\models\\YAMNet_SVM_optimized_label_encoder.pkl"
YAMNET_SVM_SCALER_PATH = r"E:\\models\\YAMNet_SVM_optimized_scaler.pkl"

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

USE_FINETUNED_AUDIO = True
audio_classifier = None
audio_labels = None
yamnet_model = None
yamnet_svm_model = None
yamnet_label_encoder = None
yamnet_scaler = None
DEFAULT_VIDEO_MIN_CONF = 0.80
DEFAULT_VIDEO_MIN_COUNT = 3
DEFAULT_VIDEO_MIN_AVG_CONF = 0.85
DEFAULT_AUDIO_THRESHOLD = 0.45
DEFAULT_AUDIO_SILENCE_RMS = 0.035
DEFAULT_AUDIO_MIN_TOP_SCORE = 0.48
DEFAULT_AUDIO_MIN_MARGIN = 0.05
HORSE_AUDIO_FORCE_SCORE = 0.60

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


def load_yamnet_classifier():
    global yamnet_model, yamnet_svm_model, yamnet_label_encoder, yamnet_scaler
    if not (
        os.path.exists(YAMNET_SVM_MODEL_PATH)
        and os.path.exists(YAMNET_SVM_LABEL_ENCODER_PATH)
        and os.path.exists(YAMNET_SVM_SCALER_PATH)
    ):
        return
    yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
    with open(YAMNET_SVM_MODEL_PATH, "rb") as f:
        yamnet_svm_model = pickle.load(f)
    with open(YAMNET_SVM_LABEL_ENCODER_PATH, "rb") as f:
        yamnet_label_encoder = pickle.load(f)
    with open(YAMNET_SVM_SCALER_PATH, "rb") as f:
        yamnet_scaler = pickle.load(f)


load_yamnet_classifier()

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

AUDIO_ALLOWED_ANIMALS = {
    "lion",
    "tiger",
    "cheetah",
    "cat",
    "dog",
    "cow",
    "deer",
    "leopard",
    "bear",
    "elephant",
    "horse",
    "wild boar",
    "boar",
    "wolf",
    "panther",
    "crocodile",
    "rhino",
    "hippo",
    "snake",
}


def is_allowed_audio_animal(label):
    normalized = str(label).strip().lower()
    return normalized not in {"human", "dog"}

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


def load_audio_file_to_waveform(file_path, target_sr=32000):
    waveform, sample_rate = librosa.load(file_path, sr=None, mono=True)
    if sample_rate != target_sr:
        waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=target_sr)
    # Ensure fixed length (3 seconds default)
    target_len = target_sr * 3
    if len(waveform) < target_len:
        waveform = np.pad(waveform, (0, target_len - len(waveform)))
    elif len(waveform) > target_len:
        waveform = waveform[:target_len]
    return waveform

def infer_audio(waveform):
    if (
        yamnet_model is not None
        and yamnet_svm_model is not None
        and yamnet_label_encoder is not None
        and yamnet_scaler is not None
    ):
        waveform_16k = librosa.resample(waveform, orig_sr=32000, target_sr=16000).astype(np.float32)
        scores, embeddings, spectrogram = yamnet_model(waveform_16k)
        embeddings_np = embeddings.numpy().astype(np.float32)
        pooled_feature = np.concatenate(
            [np.mean(embeddings_np, axis=0), np.std(embeddings_np, axis=0)],
            axis=0,
        ).reshape(1, -1)
        scaled_feature = yamnet_scaler.transform(pooled_feature)
        probs = yamnet_svm_model.predict_proba(scaled_feature)[0]
        labels = yamnet_label_encoder.inverse_transform(np.arange(len(probs)))
        return {str(label): float(prob) for label, prob in zip(labels, probs)}

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


def build_zero_scores():
    if audio_labels is not None:
        return {
            label: 0.0
            for label in (audio_labels[str(i)] for i in range(len(audio_labels)))
            if is_allowed_audio_animal(label)
        }
    return {}


def filter_audio_scores(raw_scores, threshold, min_top_score, min_margin):
    scores = {
        str(label): float(score)
        for label, score in raw_scores.items()
        if is_allowed_audio_animal(label)
    }
    if not scores:
        return [], {}

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    top_label, top_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = top_score - second_score

    if str(top_label).strip().lower() == "horse" and top_score >= HORSE_AUDIO_FORCE_SCORE:
        filtered_scores = {
            label: (score if label == top_label else 0.0)
            for label, score in scores.items()
        }
        return [top_label], filtered_scores

    if top_score < min_top_score:
        return [], {label: 0.0 for label in scores}

    if margin < min_margin and top_score < (min_top_score + 0.15):
        return [], {label: 0.0 for label in scores}

    detected = [top_label] if top_score >= threshold else []
    filtered_scores = {
        label: (score if label in detected else 0.0)
        for label, score in scores.items()
    }
    return detected, filtered_scores


def log_audio_debug(raw_scores, detected):
    if not raw_scores:
        print("[audio] no raw scores produced")
        return
    ranked = sorted(
        (
            (str(label), float(score))
            for label, score in raw_scores.items()
            if is_allowed_audio_animal(label)
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    if not ranked:
        print("[audio] no allowed audio labels after filtering")
        return
    top_preview = ", ".join(f"{label}={score:.3f}" for label, score in ranked[:5])
    if detected:
        print(f"[audio] detected={detected} | top_scores: {top_preview}")
    else:
        print(f"[audio] no confident detection | top_scores: {top_preview}")

# Local fallback video (kept for testing if IP cam is down)
video_filename = "LionM.mp4"
video_path = os.path.join("static", video_filename)

# Dangerous animals list (used only for DB flag)
DANGEROUS_ANIMALS = [
    "tiger", "leopard", "lion", "bear", "elephant",
    "wild boar", "boar", "wolf", "panther", "crocodile",
    "rhino", "hippo", "snake"
]

def is_dangerous_animal(animal_name):
    return any(dangerous in str(animal_name).lower() for dangerous in DANGEROUS_ANIMALS)

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
    class_max_conf = {}
    class_conf_sums = {}
    min_conf = float(request.args.get("min_conf", DEFAULT_VIDEO_MIN_CONF))
    min_count = int(request.args.get("min_count", DEFAULT_VIDEO_MIN_COUNT))
    min_avg_conf = float(request.args.get("min_avg_conf", DEFAULT_VIDEO_MIN_AVG_CONF))

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
            conf = float(box.conf[0]) if hasattr(box, "conf") else 1.0
            if conf < min_conf:
                continue
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            detected_classes.add(label)
            class_counts[label] = class_counts.get(label, 0) + 1
            class_max_conf[label] = max(class_max_conf.get(label, 0.0), conf)
            class_conf_sums[label] = class_conf_sums.get(label, 0.0) + conf

        frame_count += 1

    cap.release()
    if frame_count == 0:
        return jsonify({
            "status": "error",
            "message": "No frames received from IP camera stream: " + ip_camera_url
        }), 500

    # Filter out single-frame flickers/false positives
    filtered_counts = {k: v for k, v in class_counts.items() if v >= min_count}
    filtered_counts = {k: v for k, v in filtered_counts.items() if class_max_conf.get(k, 0.0) >= min_conf}
    filtered_counts = {
        k: v for k, v in filtered_counts.items()
        if (class_conf_sums.get(k, 0.0) / max(1, class_counts.get(k, 1))) >= min_avg_conf
    }
    video_scores = {label: count / frame_count for label, count in filtered_counts.items()}

    # Store detections in MongoDB (does not affect detection output)
    for animal, score in video_scores.items():
        is_dangerous = is_dangerous_animal(animal)
        store_detection(animal, "video", score, is_dangerous)
        if is_dangerous:
            send_danger_alert_email(animal, "video", score)

    return jsonify({
        "status": "success",
        "detected": list(filtered_counts.keys()),
        "counts": filtered_counts,
        "scores": video_scores,
        "frames": frame_count
    })

@app.route("/audio_detect", methods=["GET"])
def detect_from_audio():
    seconds = int(request.args.get("seconds", 4))
    threshold = float(request.args.get("threshold", DEFAULT_AUDIO_THRESHOLD))
    silence_rms = float(request.args.get("silence_rms", DEFAULT_AUDIO_SILENCE_RMS))
    min_top_score = float(request.args.get("min_top_score", DEFAULT_AUDIO_MIN_TOP_SCORE))
    min_margin = float(request.args.get("min_margin", DEFAULT_AUDIO_MIN_MARGIN))
    if not ip_camera_audio_url:
        return jsonify({
            "status": "success",
            "detected": [],
            "scores": build_zero_scores(),
            "message": "IP camera audio stream is not configured"
        })
    try:
        wav_path = capture_audio_wav(ip_camera_audio_url, seconds=seconds, sample_rate=32000)
        sample_rate, waveform = load_waveform(wav_path)
        os.unlink(wav_path)
        if sample_rate != 32000:
            waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=32000)
        # Pad/crop to requested seconds for faster inference
        target_len = 32000 * seconds
        if len(waveform) < target_len:
            waveform = np.pad(waveform, (0, target_len - len(waveform)))
        elif len(waveform) > target_len:
            waveform = waveform[:target_len]

        # Silence gate: when near-silent, return zero scores
        rms = float(np.sqrt(np.mean(np.square(waveform))))
        if rms < silence_rms:
            return jsonify({
                "status": "success",
                "detected": [],
                "scores": build_zero_scores()
            })

        raw_scores = infer_audio(waveform)
        detected_sorted, scores = filter_audio_scores(
            raw_scores,
            threshold=threshold,
            min_top_score=min_top_score,
            min_margin=min_margin,
        )
        log_audio_debug(raw_scores, detected_sorted)
        if not detected_sorted:
            return jsonify({
                "status": "success",
                "detected": [],
                "scores": scores if scores else build_zero_scores()
            })

        # Store detections in MongoDB (does not affect detection output)
        for animal in detected_sorted:
            is_dangerous = is_dangerous_animal(animal)
            score = scores.get(animal, 0.0)
            store_detection(animal, "audio", score, is_dangerous)
            if is_dangerous:
                send_danger_alert_email(animal, "audio", score)

        return jsonify({
            "status": "success",
            "detected": detected_sorted,
            "scores": scores
        })
    except Exception as e:
        print(f"[audio] ip camera audio unavailable: {e}")
        return jsonify({
            "status": "success",
            "detected": [],
            "scores": build_zero_scores(),
            "message": "IP camera audio unavailable"
        })


@app.route("/predict-audio", methods=["POST"])
def predict_audio():
    if "file" not in request.files:
        return jsonify({"status": "error", "message": "No file provided"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"status": "error", "message": "No file selected"}), 400

    try:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp.close()
        file.save(tmp.name)
        waveform = load_audio_file_to_waveform(tmp.name, target_sr=32000)
        os.unlink(tmp.name)

        raw_scores = infer_audio(waveform)
        detected, scores = filter_audio_scores(
            raw_scores,
            threshold=DEFAULT_AUDIO_THRESHOLD,
            min_top_score=DEFAULT_AUDIO_MIN_TOP_SCORE,
            min_margin=DEFAULT_AUDIO_MIN_MARGIN,
        )
        log_audio_debug(raw_scores, detected)
        if not scores:
            return jsonify({"status": "error", "message": "No scores produced"}), 500
        if not detected:
            return jsonify({"animal": "No confident animal detected", "confidence": 0.0})

        animal = detected[0]
        confidence = round(scores[animal] * 100, 2)
        store_detection(animal, "audio_file", scores[animal], False)

        return jsonify({"animal": animal, "confidence": confidence})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# Route to serve local video file (optional fallback)
@app.route("/video")
def serve_video():
    return send_from_directory("static", video_filename)

# Route to get detection history
@app.route("/history", methods=["GET"])
def get_history():
    try:
        limit = int(request.args.get("limit", 5))
        history = get_detection_history(limit)
        return jsonify({"status": "success", "history": history})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    # Disable reloader to avoid Windows watchdog/socket issues with large ML deps
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
