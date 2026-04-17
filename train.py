"""Training script to fine-tune EfficientNet-B0 for deepfake face classification."""

import os
import timm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from model import DeepFakeDetector

DATASET_PATH = "data/"
NUM_EPOCHS = 5
BATCH_SIZE = 16
LEARNING_RATE = 1e-4

# --- STRICT GPU ENFORCEMENT ---
if not torch.cuda.is_available():
    raise RuntimeError("\n" + "!"*50 + "\nFATAL ERROR: GPU (CUDA) NOT DETECTED!\n" + 
                       "Training on CPU is disabled to prevent slow runs.\n" +
                       "Please install the GPU version of PyTorch.\n" + "!"*50)

DEVICE = torch.device("cuda")
print(f"GPU detected: {torch.cuda.get_device_name(0)}")

train_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def main():
    """Train classifier on prepared real/fake face dataset and save weights."""
    
    # Check if dataset exists
    train_path = os.path.join(DATASET_PATH, "train")
    if not os.path.exists(train_path):
        print(f"\nERROR: Dataset folder not found at {train_path}")
        print("Please run: python extract_dataset_frames.py first to prepare the data.\n")
        return

    print(f"Loading dataset from {train_path}...")
    try:
        train_dataset = ImageFolder(train_path, transform=train_transform)
    except Exception as e:
        print(f"ERROR loading dataset: {e}")
        return

    class_to_idx = train_dataset.class_to_idx
    print(f"Detected classes: {class_to_idx}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    print(f"Using device: {DEVICE}")
    model = DeepFakeDetector().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

    # --- RESUME LOGIC ---
    os.makedirs("model_weights", exist_ok=True)
    checkpoint_path = "model_weights/checkpoint.pth"
    start_epoch = 0

    if os.path.exists(checkpoint_path):
        print(f"Found checkpoint at {checkpoint_path}. Resuming...")
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from Epoch {start_epoch + 1}")

    # Label mapping check
    if "real" in class_to_idx and "fake" in class_to_idx:
        max_idx = max(class_to_idx.values())
        label_map = torch.arange(max_idx + 1, dtype=torch.long)
        label_map[class_to_idx["real"]] = 0
        label_map[class_to_idx["fake"]] = 1
    else:
        label_map = None

    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        for inputs, labels in pbar:
            inputs = inputs.to(DEVICE)
            if label_map is not None:
                labels = label_map[labels]
            labels = labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        scheduler.step()
        epoch_loss = running_loss / max(len(train_loader), 1)
        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}] Finished. Avg Loss: {epoch_loss:.4f}")

        # Save Checkpoint after every epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    # Final Save
    torch.save(model.state_dict(), "model_weights/deepfake_detector.pth")
    # Optional: remove checkpoint after successful completion
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        
    print("\nTraining Complete! Final model saved to model_weights/deepfake_detector.pth")


if __name__ == "__main__":
    main()
