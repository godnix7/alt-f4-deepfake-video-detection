# DeepFake Video Detection System

## 🛡️ Overview
The DeepFake Video Detection System is an AI-driven web application designed to identify face-swap deepfakes. By extracting frames and analyzing faces using state-of-the-art neural networks, it provides users with an instant, reliable verdict on video authenticity.

## ✨ Features
- **Automated Analysis**: Extracts and processes 20 frames per video automatically.
- **Advanced Vision Models**: Powered by MTCNN for face detection and EfficientNet-B0 for classification.
- **Intuitive UI**: Simple Flask web interface for uploading and analyzing videos.
- **Confidence Scoring**: Delivers aggregated results with a clear confidence metric.

## 🛠️ Tech Stack
- **Backend & ML**: Python 3.10, PyTorch, Flask
- **Computer Vision**: OpenCV, MTCNN, EfficientNet-B0

## 🚀 Installation & Usage
1. **Clone the repository**:
   ```bash
   git clone https://github.com/godnix7/alt-f4-deepfake-video-detection.git
   cd alt-f4-deepfake-video-detection
   ```
2. **Set up the virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Create required directories**:
   ```bash
   mkdir uploads model_weights
   ```
5. **Run the application**:
   ```bash
   python app.py
   ```
   Navigate to `http://localhost:5000` in your browser.

## 🔮 Future Improvements
- Support for detecting audio deepfakes.
- Optimization for real-time video stream analysis.
- Implementation of a REST API for third-party integrations.
