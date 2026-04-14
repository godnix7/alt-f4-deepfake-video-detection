"""Model definition and loading utilities for deepfake classification."""

import os

import timm
import torch
import torch.nn as nn


class DeepFakeDetector(nn.Module):
    """EfficientNet-B0 based binary classifier for REAL vs FAKE faces."""

    def __init__(self):
        super().__init__()
        # Load pretrained EfficientNet-B0 backbone and replace the classifier head.
        model = timm.create_model("efficientnet_b0", pretrained=True)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features=1280, out_features=2),
        )
        self.model = model

    def forward(self, x):
        """Run a forward pass on input batch tensor."""
        return self.model(x)


def load_model(weights_path, device):
    """Load model weights if available and return model in eval mode."""
    model = DeepFakeDetector().to(device)
    if os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print("No weights found. Using untrained model.")
    model.eval()
    return model
