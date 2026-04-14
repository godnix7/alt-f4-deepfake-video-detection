"""Training script to fine-tune EfficientNet-B0 for deepfake face classification."""

# NOTE: Dataset structure expected:
# data/
# ├── train/
# │   ├── real/   (real face images/frames, jpg/png)
# │   └── fake/   (deepfake face images/frames, jpg/png)
# Download FaceForensics++ or use DFDC dataset frames
# For quick demo: use any small set of real vs fake face crops

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
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    train_dataset = ImageFolder(os.path.join(DATASET_PATH, "train"), transform=train_transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
    )

    model = DeepFakeDetector().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()
        epoch_loss = running_loss / max(len(train_loader), 1)
        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}] Loss: {epoch_loss:.4f}")

    os.makedirs("model_weights", exist_ok=True)
    torch.save(model.state_dict(), "model_weights/deepfake_detector.pth")
    print("Model saved!")


if __name__ == "__main__":
    main()
