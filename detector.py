"""Core inference pipeline to analyze uploaded videos for deepfake content."""

import os

import torch

from model import load_model
from utils import (
    aggregate_predictions,
    detect_and_crop_faces,
    extract_frames,
    preprocess_faces,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS_PATH = os.path.join("model_weights", "deepfake_detector.pth")
FAKE_CLASS_INDEX = int(os.getenv("FAKE_CLASS_INDEX", "1"))

# Load model once at startup for faster repeated predictions.
model = load_model(WEIGHTS_PATH, DEVICE)


def analyze_video(video_path, num_frames=20):
    """Run end-to-end video analysis and return result dictionary."""
    try:
        frames = extract_frames(video_path, num_frames)
        if not frames:
            raise ValueError("No frames could be extracted from the video.")

        face_crops = detect_and_crop_faces(frames)
        input_tensor = preprocess_faces(face_crops).to(DEVICE)

        with torch.no_grad():
            logits = model(input_tensor)

        result = aggregate_predictions(logits, fake_class_index=FAKE_CLASS_INDEX)
        return result
    except Exception as e:
        return {
            "label": "ERROR",
            "confidence": 0,
            "error": str(e),
            "frames_analyzed": 0,
        }
