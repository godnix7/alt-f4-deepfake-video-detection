"""Flask web app for uploading videos and returning deepfake predictions."""

import os
import uuid

from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

from detector import analyze_video

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024

# Ensure runtime folders exist so the app never fails on missing directories.
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("model_weights", exist_ok=True)


def allowed_file(filename):
    """Validate extension for uploaded video files."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    """Render single-page upload interface."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Handle uploaded video, run analysis, and return JSON response."""
    if "video" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    unique_filename = f"{uuid.uuid4()}_{secure_filename(file.filename)}"
    video_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)

    try:
        file.save(video_path)
        result = analyze_video(video_path)
        return jsonify(result)
    except Exception as e:
        return jsonify({"label": "ERROR", "confidence": 0, "error": str(e), "frames_analyzed": 0}), 500
    finally:
        if os.path.exists(video_path):
            os.remove(video_path)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
