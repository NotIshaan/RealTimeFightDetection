# Violence Detection Model (VDM)

A deep learning model for detecting violence in videos using transfer learning with InceptionV3. The model analyzes video frames to classify content as violent or non-violent with high accuracy.

## 🎯 Overview

This project implements a binary classification system that can identify violent content in videos by:
- Extracting frames from videos at regular intervals
- Using a pre-trained InceptionV3 model with custom classification layers
- Averaging predictions across multiple frames for robust video-level classification

## 📊 Performance

- **Accuracy**: 97% on test dataset
- **Precision**: 96-98% for both classes
- **Recall**: 96-98% for both classes
- **F1-Score**: 97% for both classes

## 🏗️ Architecture

- **Base Model**: InceptionV3 (pre-trained on ImageNet)
- **Input Size**: 299×299 RGB images
- **Custom Layers**: 
  - Global Average Pooling 2D
  - Dense layer (1024 units, ReLU activation)
  - Output layer (1 unit, Sigmoid activation)
- **Training Strategy**: Transfer learning with frozen base layers

## 📂 Dataset Structure
Dataset Link - https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset
```
Real Life Violence Dataset/
├── Violence/           # Violent video samples
└── NonViolence/        # Non-violent video samples
```

The dataset contains 2000 videos (1000 violent, 1000 non-violent) which are processed to extract frames for training.

## 🚀 Usage

### 1. Frame Extraction

```python
# Extract frames from videos
def extract_frames(video_path, output_folder, num_frames=15, resize_shape=(299, 299)):
    # Extracts evenly spaced frames from video
    # Resizes to InceptionV3 input size
```

### 2. Data Preparation

```python
# Split data into train/validation sets (80/20 split)
train_generator = train_datagen.flow_from_directory(
    '/path/to/train',
    target_size=(299, 299),
    batch_size=32,
    class_mode='binary'
)
```

### 3. Model Training

```python
# Load and configure InceptionV3
base_model = InceptionV3(weights='imagenet', include_top=False)
# Add custom classification layers
# Train for 15 epochs with early stopping
```

### 4. Video Inference

```python
# Load model and predict on new video
model = load_model('VDM_v2.keras')
frames = extract_frames_from_video(video_path, num_frames=32)
preds = model.predict(frames)
avg_pred = np.mean(preds)
label = "Violent" if avg_pred > 0.5 else "Non-Violent"
```

## 📋 Requirements

```
tensorflow>=2.8.0
opencv-python>=4.5.0
numpy>=1.21.0
scikit-learn>=1.0.0
tqdm>=4.62.0
Pillow>=8.0.0
```

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/NotIshaan/RealTimeFightDetection.git
cd RealTimeFightDetection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the dataset and place it in the appropriate directory structure.

## 📖 Training Process

1. **Frame Extraction**: Extract 15 evenly-spaced frames from each video
2. **Data Augmentation**: Apply horizontal flip and rotation for training data
3. **Transfer Learning**: Use frozen InceptionV3 features with custom classifier
4. **Training**: 15 epochs with early stopping and model checkpointing
5. **Validation**: 20% of data held out for validation

## 🔍 Model Details

- **Total Parameters**: ~22M (most frozen from InceptionV3)
- **Trainable Parameters**: ~1M (custom classification layers)
- **Optimizer**: Adam (learning rate: 0.0001)
- **Loss Function**: Binary crossentropy
- **Callbacks**: Early stopping, model checkpointing

## 📈 Training Results

| Epoch | Training Accuracy | Validation Accuracy | Training Loss | Validation Loss |
|-------|------------------|-------------------|---------------|-----------------|
| 1     | 86.14%          | 93.20%           | 0.3156        | 0.1716         |
| 5     | 96.89%          | 96.52%           | 0.0929        | 0.0911         |
| 10    | 98.15%          | 97.40%           | 0.0548        | 0.0684         |
| 15    | 98.91%          | 98.15%           | 0.0336        | 0.0515         |

## 🧪 Evaluation

The model was evaluated on 100 test videos (50 from each class):

```
Confusion Matrix:
[[48  2]
 [ 1 49]]

Classification Report:
              precision    recall  f1-score   support
 Non-Violent       0.98      0.96      0.97        50
     Violent       0.96      0.98      0.97        50
    accuracy                           0.97       100
```
I have also evaluated the model on videos directly sourced from YouTube, it performs well on most standard quality videos(720p+).
## 🎬 Example Usage

```python
# Single video prediction
video_path = "path/to/your/video.mp4"
frames = extract_frames_from_video(video_path, num_frames=32)
frames = preprocess_input(frames.astype(np.float32))
preds = model.predict(frames)
avg_pred = np.mean(preds)
confidence = avg_pred if avg_pred > 0.5 else 1 - avg_pred
label = "Violent" if avg_pred > 0.5 else "Non-Violent"
print(f"Prediction: {label} (confidence: {confidence:.2f})")
```

## ⚠️ Limitations

- Model performance depends on video quality and frame extraction
- Designed for binary classification (violent/non-violent)
- May require retraining for specific domains or different types of violence
- Processing time scales with video length and number of frames extracted

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- InceptionV3 model from TensorFlow/Keras
- Real Life Violence Dataset contributors
- Transfer learning techniques from the deep learning community

## 📞 Contact

For questions or suggestions, please open an issue or contact [ishaanworks24@gmail.com].

---

⭐ If you find this project helpful, please consider giving it a star!
