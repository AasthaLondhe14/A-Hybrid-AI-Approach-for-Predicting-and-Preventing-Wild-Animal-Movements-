from flask import Flask, jsonify, send_from_directory
from ultralytics import YOLO
import cv2
import os
from flask_cors import CORS

app = Flask(__name__, static_folder="static")
CORS(app)

# Load YOLOv8 model
model = YOLO(os.path.join("yolov8", "best.pt"))

# IP camera stream (update if your camera uses a different path)
ip_camera_url = "http://192.0.0.4:8080/video"

# Local fallback video (kept for testing if IP cam is down)
video_filename = "LionM.mp4"
video_path = os.path.join("static", video_filename)

@app.route("/detect", methods=["GET"])
def detect_from_video():
    # Read directly from the IP camera stream
    cap = cv2.VideoCapture(ip_camera_url)
    detected_classes = set()
    class_counts = {}

    if not cap.isOpened():
        return jsonify({"status": "error", "message": "Could not open IP camera stream"}), 500

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

# Route to serve local video file (optional fallback)
@app.route("/video")
def serve_video():
    return send_from_directory("static", video_filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
