---
title: Plant Disease Detector
emoji: 🌿
colorFrom: green
colorTo: blue
sdk: streamlit
sdk_version: 1.35.0
app_file: app/app.py
pinned: false
---
# AI Plant Disease Detector

A deep learning-based web application for detecting and classifying plant diseases from leaf images using transfer learning with MobileNetV2.

The project was trained on the PlantVillage dataset and deployed as a real-time Streamlit application for practical inference and disease diagnosis.

---

## Live Demo

* Live App: [AI Plant Disease Detector Live App](https://plant-disease-detector-rumman.streamlit.app/?utm_source=chatgpt.com)
* Weights & Biases Experiments: [W&B Training Dashboard](https://wandb.ai/uman66-meta/plant-disease-detector/workspace?nw=nwuseruman66&utm_source=chatgpt.com)
* Dataset: [PlantVillage Dataset on Kaggle](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset?utm_source=chatgpt.com)

---

# Project Overview

This project classifies plant leaf diseases across 38 classes using a transfer learning pipeline built with TensorFlow and MobileNetV2.

The system includes:

* Transfer learning with ImageNet pretrained weights
* Real-time disease prediction web application
* Data augmentation pipeline
* Class imbalance handling using class weights
* Experiment tracking with Weights & Biases
* Baseline CNN vs Transfer Learning comparison
* Streamlit deployment pipeline
* Confidence-based inference system
* Treatment recommendation UI

The model was trained on GPU using Google Colab.

---

# Model Performance

## Final Transfer Learning Model

| Metric              | Value              |
| ------------------- | ------------------ |
| Validation Accuracy | 96.66%             |
| Training Accuracy   | 96.92%             |
| Validation Loss     | 0.0995             |
| Classes             | 38                 |
| Input Size          | 224 × 224          |
| Epochs              | 20                 |
| Framework           | TensorFlow / Keras |

### Classification Report Summary

| Metric            | Score |
| ----------------- | ----- |
| Accuracy          | 96%   |
| Macro F1 Score    | 0.95  |
| Weighted F1 Score | 0.96  |

---

# Transfer Learning Architecture

The final model uses MobileNetV2 pretrained on ImageNet as the feature extractor.

## Architecture

```python
Input (224x224x3)
│
├── Data Augmentation
├── MobileNetV2 (ImageNet pretrained)
├── GlobalAveragePooling2D
├── Dropout
├── Dense(128, ReLU)
├── Dropout
└── Dense(38, Softmax)
```

## Model Summary

| Layer                  | Output Shape |
| ---------------------- | ------------ |
| MobileNetV2            | (7, 7, 1280) |
| GlobalAveragePooling2D | (1280)       |
| Dense Layer            | (128)        |
| Output Layer           | (38)         |

### Parameters

| Type                 | Count     |
| -------------------- | --------- |
| Total Parameters     | 2,426,854 |
| Trainable Parameters | 894,310   |
| Model Size           | 9.26 MB   |

---

# Dataset Information

The project uses the PlantVillage dataset.

## Dataset Details

| Property           | Value           |
| ------------------ | --------------- |
| Total Classes      | 38              |
| Image Shape        | 256×256×3       |
| Dataset Type       | RGB Leaf Images |
| Mean Pixel Value   | 0.4623          |
| Standard Deviation | 0.1842          |
| Imbalance Ratio    | 36.23           |

Only the `color` images folder from the dataset was used during training.

---

# Data Augmentation

To improve generalization and reduce overfitting, an augmentation pipeline was added directly into the model.

```python
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
])
```

---

# Class Imbalance Handling

The dataset contained significant imbalance between classes.

To address this issue, class weights were computed dynamically using Scikit-learn.

```python
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y),
    y=y
)
```

This improved minority class learning stability and reduced bias toward dominant classes.

---

# Training Strategy

## Callbacks Used

### Early Stopping

Stops training when validation loss stops improving.

```python
EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)
```

### ReduceLROnPlateau

Reduces learning rate automatically during plateaus.

```python
ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,
    patience=2,
    min_lr=1e-7
)
```

### Model Checkpointing

Saves the best-performing model during training.

```python
ModelCheckpoint(
    "models/best_model.keras",
    monitor="val_accuracy",
    save_best_only=True
)
```

---

# Baseline CNN vs Transfer Learning

A custom CNN was first trained as a baseline model.

## Baseline CNN Results

| Metric              | Value |
| ------------------- | ----- |
| Validation Accuracy | 84.2% |
| Weighted F1 Score   | 0.84  |

The CNN struggled heavily with visually similar tomato disease classes and minority categories.

## Worst Performing CNN Classes

| Class                     | Recall |
| ------------------------- | ------ |
| Tomato Late Blight        | 48.5%  |
| Tomato Septoria Leaf Spot | 54.8%  |
| Pepper Bacterial Spot     | 56.9%  |
| Apple Scab                | 59.2%  |
| Tomato Target Spot        | 65.2%  |

---

## Transfer Learning Improvements

After switching to MobileNetV2 transfer learning:

| Metric                     | CNN             | MobileNetV2                  |
| -------------------------- | --------------- | ---------------------------- |
| Accuracy                   | 84%             | 96%                          |
| Generalization             | Moderate        | Strong                       |
| Minority Class Performance | Weak            | Improved                     |
| Feature Extraction         | Manual Learning | Pretrained ImageNet Features |

The transfer learning approach significantly improved robustness and disease discrimination capability.

---

# Worst Performing Classes (Final Model)

Despite high overall accuracy, several visually similar disease categories remained challenging.

| Class                     | Recall |
| ------------------------- | ------ |
| Tomato Early Blight       | 69.6%  |
| Tomato Septoria Leaf Spot | 84.8%  |
| Corn Gray Leaf Spot       | 86.4%  |
| Tomato Target Spot        | 86.8%  |
| Tomato Late Blight        | 90.0%  |

This behavior is expected because many tomato diseases share overlapping texture and lesion patterns.

---

# Web Application Features

The Streamlit application includes:

* Real-time image prediction
* Responsive modern UI
* Confidence score visualization
* Top-3 predictions display
* Disease treatment recommendations
* Healthy leaf detection
* Low-confidence warnings
* Cached model loading for faster inference

---

# Tech Stack

## Machine Learning

* TensorFlow
* Scikit-learn
* NumPy

## Computer Vision

* Pillow
* OpenCV

## Visualization

* Matplotlib
* Plotly

## Deployment

* Streamlit

## Reporting

* ReportLab

---

# Project Structure

```bash
plant-disease-detector/
│
├── app/              # Streamlit application
├── models/           # Saved trained models
├── notebooks/        # Training and experimentation notebooks
├── plots/            # Generated evaluation plots
├── data/             # Dataset storage
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/Uman-66/plant-disease-detector.git
cd plant-disease-detector
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run Streamlit App

```bash
streamlit run app/app.py
```

---

# Future Improvements

Potential future improvements include:

* Grad-CAM visual explanations
* Multi-leaf disease detection
* Better handling of visually similar tomato diseases
* Quantization for edge/mobile deployment
* ONNX or TensorRT optimization
* Disease severity estimation
* Mobile application deployment
* Ensemble learning approaches

---

# Key Learning Outcomes

This project involved practical experience with:

* Transfer learning
* Deep learning optimization
* Class imbalance handling
* ML experiment tracking
* Real-world deployment
* Model evaluation and error analysis
* Streamlit application engineering
* Production-oriented inference pipelines

---

# Notes

This project is intended for educational and portfolio purposes and should not replace professional agricultural diagnosis in real-world farming environments.
