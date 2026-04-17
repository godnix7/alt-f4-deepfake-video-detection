"""Utility functions for video frames, face crops, preprocessing, and aggregation."""

import cv2
import numpy as np
import torch
from PIL import Image
from facenet_pytorch import MTCNN
from torchvision import transforms


def extract_frames(video_path, num_frames=20):
    """Extract evenly spaced RGB frames from a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Cannot open video file")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []

    frame_indices = np.linspace(0, max(total_frames - 1, 0), num=num_frames, dtype=int)
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        success, frame = cap.read()
        if not success or frame is None:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    cap.release()
    return frames


def detect_and_crop_faces(frames, image_size=224):
    """Detect face crops using MTCNN, with center-crop fallback when needed."""
    mtcnn = MTCNN(
        image_size=image_size,
        margin=20,
        keep_all=False,
        post_process=False,
        device="cpu",
    )

    face_crops = []
    for frame in frames:
        pil_img = Image.fromarray(frame)
        face_tensor = mtcnn(pil_img)
        if face_tensor is not None:
            face_crops.append(face_tensor)

    # Fallback: center-crop resized frame if no faces are detected.
    if not face_crops:
        for frame in frames:
            pil_img = Image.fromarray(frame)
            width, height = pil_img.size
            crop_size = min(width, height)
            left = (width - crop_size) // 2
            top = (height - crop_size) // 2
            right = left + crop_size
            bottom = top + crop_size
            face_crops.append(pil_img.crop((left, top, right, bottom)).resize((image_size, image_size)))

    return face_crops


def preprocess_faces(face_list):
    """Convert detected faces to normalized model input tensor batch."""
    if not face_list:
        raise ValueError("No valid frames/faces available for analysis.")

    normalize_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    to_pil = transforms.ToPILImage()
    tensor_list = []
    for face in face_list:
        if isinstance(face, torch.Tensor):
            # MTCNN tensors may be in [0,255] (post_process=False) or [-1,1].
            face = face.detach().cpu().float()
            if face.numel() == 0:
                continue
            if float(face.max()) > 1.0:
                face = face / 255.0
            if float(face.min()) < 0.0:
                face = (face + 1.0) / 2.0
            face = torch.clamp(face, 0.0, 1.0)
            face = to_pil(face)
        tensor_list.append(normalize_transform(face))

    if not tensor_list:
        raise ValueError("No valid face crops after preprocessing.")

    return torch.stack(tensor_list)


def aggregate_predictions(logits_tensor, fake_class_index=1):
    """Aggregate frame logits into a single video-level verdict and confidence."""
    if fake_class_index not in (0, 1):
        raise ValueError("fake_class_index must be 0 or 1.")

    probs = torch.softmax(logits_tensor, dim=1)
    fake_probs = probs[:, fake_class_index]
    mean_fake_prob = torch.mean(fake_probs)

    if mean_fake_prob >= 0.5:
        label = "FAKE"
        confidence = mean_fake_prob
    else:
        label = "REAL"
        confidence = 1 - mean_fake_prob

    return {
        "label": label,
        "confidence": round(float(confidence) * 100, 2),
        "fake_prob": round(float(mean_fake_prob) * 100, 2),
        "frames_analyzed": int(logits_tensor.shape[0]),
    }
