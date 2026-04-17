import os
import cv2
import torch
from tqdm import tqdm
from utils import extract_frames, detect_and_crop_faces
from PIL import Image

# Configuration
SOURCE_BASE = "data"
TARGET_BASE = "data/train"
FRAMES_PER_VIDEO = 10

def process_videos(source_dir, label, num_frames=5):
    if not os.path.exists(source_dir):
        print(f"Skipping: {source_dir} (not found)")
        return

    target_dir = os.path.join(TARGET_BASE, label)
    os.makedirs(target_dir, exist_ok=True)
    
    # Recursively find all mp4 files
    video_files = []
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith(".mp4"):
                video_files.append(os.path.join(root, file))
                
    if not video_files:
        print(f"No videos found in {source_dir}")
        return

    print(f"Extracting faces from {len(video_files)} videos for '{label}'...")
    for video_path in tqdm(video_files):
        try:
            video_name = os.path.basename(video_path).replace(".mp4", "")
            # Check if already processed (skip if first frame exists)
            first_frame_path = os.path.join(target_dir, f"{video_name}_f0.jpg")
            if os.path.exists(first_frame_path):
                continue

            # Extract frames
            frames = extract_frames(video_path, num_frames=num_frames)
            if not frames:
                continue
                
            # Detect and crop faces
            face_crops = detect_and_crop_faces(frames)
            
            # Save crops
            for i, face in enumerate(face_crops):
                if isinstance(face, torch.Tensor):
                    # MTCNN returns [C, H, W] tensors in [0, 255] if post_process=False
                    face_np = face.permute(1, 2, 0).byte().cpu().numpy()
                    face_img = Image.fromarray(face_np)
                else:
                    face_img = face # Fallback is already PIL
                
                save_path = os.path.join(target_dir, f"{video_name}_f{i}.jpg")
                face_img.save(save_path)
        except Exception as e:
            # Silent fail for individual videos to keep progress moving
            continue

if __name__ == "__main__":
    # Real videos
    process_videos(os.path.join(SOURCE_BASE, "DFD_original sequences"), "real", num_frames=FRAMES_PER_VIDEO)
    # Deepfake videos
    process_videos(os.path.join(SOURCE_BASE, "DFD_manipulated_sequences"), "fake", num_frames=FRAMES_PER_VIDEO)
    
    print("\nPreprocessing complete! You can now run the training script.")
