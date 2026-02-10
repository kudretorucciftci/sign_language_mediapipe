<<<<<<< HEAD
# 🖐️ Sign Language Recognition System

A high-performance sign language recognition system powered by **MediaPipe Hands** and **Deep Learning**. This project detects 25 letters of the alphabet (excluding J) in real-time using hand landmarks.

---

## 🛠️ Technology Stack & Libraries

- **Core Logic:** Python 3.11
- **Computer Vision:** [MediaPipe](https://mediapipe.dev/) (Hand Landmark Detection)
- **Deep Learning:** [TensorFlow 2.17 / Keras 3](https://tensorflow.org/) (Neural Network)
- **Image Processing:** OpenCV
- **GUI:** Tkinter & PIL (Pillow)
- **Data Analysis:** NumPy, Matplotlib, Scikit-learn

---

## 📐 System Architecture

The project follows a modular pipeline from data acquisition to real-time inference:

### 1. Data Collection (`data_collector.py`)
Instead of raw images, the system captures **63 physical coordinates** (21 points × X, Y, Z) of the hand. This makes the model extremely lightweight (~700KB) and robust against lighting/background changes.

### 2. Dataset Preparation (`prepare_dataset.py`)
Accumulated `.npy` files are aggregated into a single vectorized dataset (`X_dataset.npy` and `y_labels.npy`).

### 3. Model Training (`train_model.py`)
A Deep Neural Network (DNN) with the following architecture:
- **Input Layer:** 63 features
- **Hidden Layers:** 256 -> 128 -> 64 with **Dropout** (0.4, 0.3, 0.2) to prevent overfitting.
- **Output Layer:** 25 classes with **Softmax** activation.
- **Optimizer:** Adam
- **Loss Function:** Sparse Categorical Crossentropy

### 4. Evaluation (`model_evaluation.py`)
Detailed performance metrics including a **Classification Report** and **Confusion Matrix** to identify specifically which letters are being confused.

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python 3.11 installed.

### 2. Installation
Install all dependencies using:
```bash
py -3.11 -m pip install -r requirements.txt
```

### 3. Running the App
The easiest way is to use the provided command file:
- Simply double-click **`start_app.cmd`**
- Or run in terminal: `py -3.11 gui_inference.py`

---

## 📂 File Structure (English Renamed)

- `data_collector.py`: Collect hand landmark data for new letters.
- `prepare_dataset.py`: Process raw coordinates into training data.
- `train_model.py`: Train the Deep Learning model.
- `gui_inference.py`: Main GUI application for real-time detection (Inference).
- `inference_script.py`: Console-based detection script.
- `model_evaluation.py`: Analyze model accuracy and confusion matrix.
- `camera_test.py`: Utility to detect available camera indices (iVCam/Built-in).
- `sign_lang_model.keras`: The pre-trained production model.
- `hand_coordinate_data/`: Directory containing raw `.npy` data.

---

## 🎯 Fine-Tuning & Customization
To teach the system new signs:
1. Run `data_collector.py` to record at least 500 samples per sign.
2. Run `prepare_dataset.py` to rebuild the matrix.
3. Run `train_model.py` to update the `.keras` model file.

---

## 🇹🇷 Türkçe Özet
Bu proje, MediaPipe ve TensorFlow kullanarak 25 harfi tanıyan bir işaret dili tanıma sistemidir. Görüntü yerine 21 eklem noktasının koordinatlarını kullandığı için çok hızlı çalışır. **`start_app.cmd`** dosyasına tıklayarak başlatabilir, "Switch Camera" butonu ile IVCam veya dahili kamera arasında geçiş yapabilirsiniz.

---
*Developed by Antigravity AI*
=======
# Sign Language Mediapipe

## Proje Hakkında
Bu proje, **Mediapipe ve yapay zeka** kullanarak **işaret dili tanıma** uygulaması geliştirmeyi amaçlamaktadır.  
Gerçek zamanlı el hareketlerini algılar ve işaret edilen harfleri tahmin eder.

---

## Kullanılan Teknolojiler
- Python  
- Mediapipe  
- TensorFlow / Keras  
- OpenCV  
- NumPy  

---

## Kurulum ve Kullanım
1. Depoyu klonlayın:
```bash
git clone https://github.com/kudretorucciftci/sign_language_mediapipe.git

>>>>>>> df429e16b6e3287b1af5a4c41112f66d5967a35c
