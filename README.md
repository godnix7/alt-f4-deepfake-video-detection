# DeepFake Video Detection System

## Overview
This project detects whether an uploaded video contains face-swap deepfake content.  
It extracts frames, detects faces, classifies each face as REAL or FAKE, and aggregates the result.  
A Flask web interface lets users upload a video and instantly view the final verdict with confidence.

## Tech Stack
- Python 3.10
- PyTorch
- EfficientNet-B0
- MTCNN
- Flask
- OpenCV

## Project Structure
```text
deepfake_detector/
|-- app.py                  # Flask web server (main entry point)
|-- detector.py             # Core detection logic
|-- model.py                # EfficientNet-B0 model definition
|-- train.py                # Training script (for reference/demo)
|-- utils.py                # Helper functions (frame extraction, face crop)
|-- templates/
|   `-- index.html          # Single page upload + result UI
|-- static/
|   `-- style.css           # Basic clean CSS
|-- uploads/                # Temp folder for uploaded videos (create this dir)
|-- model_weights/          # Folder to store trained .pth file (create this dir)
|-- requirements.txt        # All dependencies
`-- README.md               # Setup and run instructions
```

## Setup Instructions

### 1. Clone / Download the project
```bash
git clone https://github.com/godnix7/alt-f4-deepfake-video-detection
cd deepfake_detector
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Create required folders
```bash
mkdir uploads model_weights
```

### 5. (Optional) Train the model
If you have a dataset of real/fake face frames:
```bash
python train.py
```
Place the saved weights at: model_weights/deepfake_detector.pth

Note: Without weights, the app still runs using the untrained pretrained backbone
(results will be random for demo purposes).

### 6. Run the application
```bash
python app.py
```
Open browser: http://localhost:5000

## How It Works
1. User uploads a video file
2. System extracts 20 evenly-spaced frames
3. MTCNN detects and crops faces from each frame
4. EfficientNet-B0 classifies each face crop as REAL or FAKE
5. Predictions are averaged -> Final verdict with confidence score

## Dataset (for training)
Recommended datasets:
- FaceForensics++ (https://github.com/ondyari/FaceForensics)
- DFDC Dataset (Kaggle DeepFake Detection Challenge)

## Dependencies
See requirements.txt

## Known Limitations
- Without trained weights, predictions are not meaningful
- Performance depends on video quality and face visibility
- Does not handle non-face videos well

## Troubleshooting Prediction Quality
- Ensure `model_weights/deepfake_detector.pth` is actually trained on your dataset.
- If predictions look inverted, set `FAKE_CLASS_INDEX=0` before running `python app.py` and test again.
- Retrain with `python train.py` after confirming class folders are `real/` and `fake/`.
