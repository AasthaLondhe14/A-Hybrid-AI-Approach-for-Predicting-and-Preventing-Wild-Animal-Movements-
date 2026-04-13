from flask import Flask, jsonify, send_from_directory, request
from ultralytics import YOLO
import cv2
import os
import tempfile
import subprocess
import time
import wave
import numpy as np
import librosa
import urllib.request
import json
import pickle
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

# Load YOLOv8 model (video detection only)
VIDEO_MODEL_PATH = r"E:\\Large Animal Dataset\\best_Un.pt"
VIDEO_LABELS_PATH = r"E:\\Large Animal Dataset\\labels_un.json"
model = YOLO(VIDEO_MODEL_PATH)
try:
    with open(VIDEO_LABELS_PATH, "r", encoding="utf-8") as f:
        video_labels = json.load(f)
    if isinstance(video_labels, list) and video_labels:
        model.names = {i: str(name) for i, name in enumerate(video_labels)}
except Exception:
    pass

# IP camera stream (update if your camera uses a different path)
ip_camera_url = "http://100.84.198.0:8080/video"

# IP camera audio stream (IP-only; no microphone fallback)
ip_camera_audio_url = "http://100.84.198.0:8080/audio.wav"

YAMNET_SVM_MODEL_PATH = r"E:\\models\\YAMNet_SVM_optimized_model.pkl"
YAMNET_SVM_LABEL_ENCODER_PATH = r"E:\\models\\YAMNet_SVM_optimized_label_encoder.pkl"
YAMNET_SVM_SCALER_PATH = r"E:\\models\\YAMNet_SVM_optimized_scaler.pkl"
yamnet_model = None
yamnet_svm_model = None
yamnet_label_encoder = None
yamnet_scaler = None
DEFAULT_VIDEO_MIN_CONF = 0.70
DEFAULT_VIDEO_MIN_COUNT = 3
DEFAULT_VIDEO_MIN_AVG_CONF = 0.85
DEFAULT_VIDEO_MIN_MARGIN = 0.05
DEFAULT_VIDEO_MIN_TOP_CONF = 0.85
DEFAULT_VIDEO_MIN_MOTION = 1.5
VIDEO_BLANK_MEAN = 5.0
VIDEO_BLANK_STD = 5.0
VIDEO_BLANK_LAPLACIAN_VAR = 12.0
DEFAULT_AUDIO_THRESHOLD = 0.45
DEFAULT_AUDIO_SILENCE_RMS = 0.035
DEFAULT_AUDIO_ENABLE_RMS = 0.06
DEFAULT_AUDIO_ACTIVE_AMPLITUDE = 0.02
DEFAULT_AUDIO_MIN_ACTIVE_RATIO = 0.01
DEFAULT_AUDIO_MIN_TOP_SCORE = 0.48
DEFAULT_AUDIO_MIN_MARGIN = 0.05
HORSE_AUDIO_FORCE_SCORE = 0.60
ENABLE_AUDIO_DETECTION = True
AUDIO_STREAM_CHECK_TTL = 5
_last_audio_check_time = 0.0
_last_audio_check_ok = False

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

def is_allowed_audio_animal(label):
    normalized = str(label).strip().lower()
    return normalized not in {"human", "dog"}

def is_blank_frame(frame):
    try:
        if frame is None or frame.size == 0:
            return True
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean, std = cv2.meanStdDev(gray)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        return (
            float(mean[0][0]) < VIDEO_BLANK_MEAN
            and float(std[0][0]) < VIDEO_BLANK_STD
            and lap_var < VIDEO_BLANK_LAPLACIAN_VAR
        )
    except Exception:
        return True

def is_audio_stream_available(url):
    global _last_audio_check_time, _last_audio_check_ok
    now = time.time()
    if now - _last_audio_check_time < AUDIO_STREAM_CHECK_TTL:
        return _last_audio_check_ok
    _last_audio_check_time = now
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            chunk = response.read(64)
            _last_audio_check_ok = len(chunk) > 0
    except Exception:
        _last_audio_check_ok = False
    return _last_audio_check_ok
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

    return {}


def build_zero_scores():
    if yamnet_label_encoder is None:
        return {}
    try:
        labels = list(getattr(yamnet_label_encoder, "classes_", []))
    except Exception:
        labels = []
    return {str(label): 0.0 for label in labels if is_allowed_audio_animal(label)}


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
    min_margin = float(request.args.get("min_margin", DEFAULT_VIDEO_MIN_MARGIN))
    min_top_conf = float(request.args.get("min_top_conf", DEFAULT_VIDEO_MIN_TOP_CONF))
    min_motion = float(request.args.get("min_motion", DEFAULT_VIDEO_MIN_MOTION))

    if not cap.isOpened():
        return jsonify({
            "status": "error",
            "message": "Could not open IP camera stream. Check the URL: " + ip_camera_url
        }), 500

    frame_count = 0
    motion_scores = []
    prev_gray = None
    max_wait_seconds = 6
    start_time = time.time()
    while frame_count < 5:  # Process only 5 frames for speed
        if time.time() - start_time > max_wait_seconds:
            break
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            motion_scores.append(float(np.mean(diff)))
        prev_gray = gray
        if is_blank_frame(frame):
            frame_count += 1
            continue
        results = model(frame)[0]
        if results is None:
            frame_count += 1
            continue
        if results.boxes is None:
            probs = getattr(results, "probs", None)
            if probs is None:
                frame_count += 1
                continue
            try:
                top_idx = int(probs.top1)
                top_conf = float(probs.top1conf)
                top5conf = getattr(probs, "top5conf", None)
                margin = None
                if top5conf is not None and len(top5conf) > 1:
                    margin = float(top5conf[0]) - float(top5conf[1])
            except Exception:
                frame_count += 1
                continue
            if (
                top_conf >= min_conf
                and top_conf >= min_top_conf
                and (margin is None or margin >= min_margin)
            ):
                if isinstance(model.names, dict):
                    label = model.names.get(top_idx, str(top_idx))
                else:
                    label = model.names[top_idx] if len(model.names) > top_idx else str(top_idx)
                detected_classes.add(label)
                class_counts[label] = class_counts.get(label, 0) + 1
                class_max_conf[label] = max(class_max_conf.get(label, 0.0), top_conf)
                class_conf_sums[label] = class_conf_sums.get(label, 0.0) + top_conf
            frame_count += 1
            continue
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

    if motion_scores and (sum(motion_scores) / len(motion_scores)) < min_motion:
        return jsonify({
            "status": "success",
            "detected": [],
            "counts": {},
            "scores": {},
            "frames": frame_count
        })

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
    if not ENABLE_AUDIO_DETECTION:
        return jsonify({
            "status": "success",
            "detected": [],
            "scores": build_zero_scores(),
            "message": "Audio detection is disabled"
        })
    seconds = int(request.args.get("seconds", 4))
    threshold = float(request.args.get("threshold", DEFAULT_AUDIO_THRESHOLD))
    silence_rms = float(request.args.get("silence_rms", DEFAULT_AUDIO_SILENCE_RMS))
    enable_rms = float(request.args.get("enable_rms", DEFAULT_AUDIO_ENABLE_RMS))
    active_amp = float(request.args.get("active_amp", DEFAULT_AUDIO_ACTIVE_AMPLITUDE))
    min_active_ratio = float(request.args.get("min_active_ratio", DEFAULT_AUDIO_MIN_ACTIVE_RATIO))
    min_top_score = float(request.args.get("min_top_score", DEFAULT_AUDIO_MIN_TOP_SCORE))
    min_margin = float(request.args.get("min_margin", DEFAULT_AUDIO_MIN_MARGIN))
    if not ip_camera_audio_url:
        return jsonify({
            "status": "success",
            "detected": [],
            "scores": build_zero_scores(),
            "message": "IP camera audio stream is not configured"
        })
    if not is_audio_stream_available(ip_camera_audio_url):
        return jsonify({
            "status": "success",
            "detected": [],
            "scores": build_zero_scores(),
            "message": "IP camera audio stream unavailable"
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
        active_ratio = float(np.mean(np.abs(waveform) > active_amp))
        if rms < silence_rms or rms < enable_rms or active_ratio < min_active_ratio:
            return jsonify({
                "status": "success",
                "detected": [],
                "scores": build_zero_scores(),
                "message": "IP camera audio muted or disabled"
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
