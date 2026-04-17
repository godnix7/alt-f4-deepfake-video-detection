"""
Unified script to extract face frames and train the model in one go.
Safe to run and leave overnight.
"""

import os
import shutil
import torch
from extract_dataset_frames import process_videos
from train import main as train_main

# --- PREPROCESSING CONFIGURATION ---
SOURCE_BASE = "data" 
TARGET_BASE = os.path.join("data", "train")
FRAMES_PER_VIDEO = 10 # Higher quality training

# Folder names in your downloaded dataset
REAL_FOLDER = "DFD_original sequences"
FAKE_FOLDER = "DFD_manipulated_sequences"

def run_pipeline():
    print("="*50)
    print("STARTING FULL DEEPFAKE TRAINING PIPELINE")
    print("="*50)

    # Ensure directories exist
    os.makedirs(os.path.join(TARGET_BASE, "real"), exist_ok=True)
    os.makedirs(os.path.join(TARGET_BASE, "fake"), exist_ok=True)

    # 2. Extract Face Frames
    print("\n[PHASE 1] Extracting face frames from videos...")
    
    # Process Real Videos
    process_videos(os.path.join(SOURCE_BASE, REAL_FOLDER), "real", num_frames=FRAMES_PER_VIDEO)
    
    # Process Fake Videos
    process_videos(os.path.join(SOURCE_BASE, FAKE_FOLDER), "fake", num_frames=FRAMES_PER_VIDEO)
    
    print("\n[PHASE 1] COMPLETED: Dataset is ready.")

    # 3. Running Training
    print("\n[PHASE 2] Starting Model Training...")
    try:
        train_main()
    except Exception as e:
        print(f"\n[PHASE 2] FAILED: {e}")
        return

    print("\n" + "="*50)
    print("ALL DONE! Model saved to model_weights/deepfake_detector.pth")
    print("="*50)

if __name__ == "__main__":
    run_pipeline()
