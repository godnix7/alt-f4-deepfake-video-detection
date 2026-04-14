# AI/ML-Based Detection of Face-Swap Deepfake Videos

## 1. Introduction
Digital video manipulation has become increasingly accessible due to advances in Generative Adversarial Networks (GANs) and autoencoder architectures. Face-swap deepfake videos, where one person's face is digitally replaced with another's, pose serious threats to personal privacy, political integrity, and public trust in digital media. This project presents a simple, working AI/ML pipeline that automatically detects whether a given video contains face-swap deepfake content using deep learning.

## 2. Problem Statement
Face-swap deepfake videos are created by training AI models to map facial features of a target person onto a source video. The rapid proliferation of such content on social media has led to misinformation campaigns, non-consensual intimate imagery, and political manipulation. In the real world, courts, media organizations, and social platforms need automated tools to flag manipulated content at scale. The core problem addressed in this work is: given an input video, determine whether it contains deepfake (AI-generated face-swapped) content, and output a confidence score.

## 3. Existing Systems
FaceForensics++ is one of the most widely used benchmarks for deepfake detection and provides manipulated video datasets and baseline models for comparison. MesoNet introduced a lightweight CNN approach focused on mesoscopic image features and became a practical early model for deepfake detection when compute resources are limited. Facebook's DeepFake Detection Challenge (DFDC) pushed the field forward by releasing a large-scale dataset and encouraging robust solutions under realistic noise and compression conditions. However, many existing systems are trained on specific datasets and do not generalize well to unseen deepfake generation methods. Their computational requirements can also be high, and many solutions are not provided as simple deployable tools with user-friendly interfaces for non-technical users.

## 4. System Design

### 4.1 Architecture Overview
The proposed system follows a straightforward sequential pipeline: Input Video -> Frame Extraction (OpenCV) -> Face Detection (MTCNN) -> Face Crops -> Feature Extraction + Classification (EfficientNet-B0) -> Frame-level Predictions -> Aggregation -> Final Verdict. This modular architecture is easy to understand and practical for academic demonstrations.

### 4.2 Tools and Technologies
- Python 3.10: Core programming language
- PyTorch: Deep learning framework for model definition and inference
- EfficientNet-B0 (via timm library): CNN backbone pretrained on ImageNet, fine-tuned for real/fake face classification. Chosen for its balance of accuracy and efficiency.
- MTCNN (Multi-task Cascaded Convolutional Networks): Face detection and alignment model. Used to isolate facial regions before classification.
- OpenCV: Video frame extraction and image processing
- Flask: Lightweight Python web framework for the user interface
- HTML/CSS: Single-page frontend for video upload and result display

### 4.3 How the System Works (step by step)
Step 1 - Frame Extraction: 20 evenly spaced frames are extracted from the video to balance coverage and processing speed.  
Step 2 - Face Detection: MTCNN identifies and crops facial regions from each frame. If no face is found, the center crop is used as fallback.  
Step 3 - Preprocessing: Each face crop is resized to 224x224 pixels and normalized using ImageNet statistics for compatibility with EfficientNet.  
Step 4 - Classification: Each preprocessed face is passed through the EfficientNet-B0 model which outputs logits for REAL and FAKE classes.  
Step 5 - Aggregation: Softmax probabilities across all frames are averaged. If the mean fake probability >= 50%, the video is labeled FAKE.  
Step 6 - Output: The system returns the verdict (REAL/FAKE), confidence percentage, and number of frames analyzed.

### 4.4 Model Details
- Base Model: EfficientNet-B0 (pretrained on ImageNet-1k)
- Modification: Classifier head replaced with Dropout(0.3) + Linear(1280->2)
- Training: Fine-tuned on deepfake face crops (real vs fake binary classification)
- Loss Function: CrossEntropyLoss
- Optimizer: Adam (lr=1e-4), StepLR scheduler

## 5. Objectives
The first objective is to build a functional, end-to-end deepfake video detection system that is simple to deploy and use. The second objective is to leverage transfer learning with EfficientNet-B0 for efficient face-level deepfake classification. Another objective is to provide a web-based interface so non-technical users can upload videos and receive instant feedback. Finally, this project aims to demonstrate that a straightforward pipeline of frame extraction, face detection, and CNN-based classification can achieve meaningful deepfake detection.

## 6. Conclusion
A complete deepfake video detection web application was successfully built. The system uses MTCNN for face isolation and EfficientNet-B0 for binary classification of face crops as real or deepfake. The Flask web interface allows any user to upload a video and receive a REAL/FAKE verdict with a confidence score in seconds. The modular design with separate detection, model, and utility modules makes the codebase easy to understand, extend, and improve.

## 7. Future Work
Future improvements include training on larger datasets such as DFDC and FaceForensics++ to improve generalization across manipulation styles. Temporal analysis can be added by integrating LSTM layers to capture frame-to-frame inconsistencies instead of classifying each frame independently. The system can be extended with audio deepfake detection to identify synthetic voice along with fake video. A production-ready REST API can also be developed so social media platforms can integrate automated moderation workflows. Explainability methods like GradCAM can be added to highlight facial regions influencing fake predictions, improving transparency. Finally, optimization for mobile deployment using ONNX or TFLite can make inference feasible on edge devices.
