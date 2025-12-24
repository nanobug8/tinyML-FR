# Access Control System: Efficient Facial Recognition for IoT

This project implements a robust access control system based on **facial recognition** using Deep Learning and Transfer Learning techniques.

The main focus is computational optimization for deployment on **Edge/IoT devices** with limited resources, achieving high accuracy without sacrificing performance.

## 📄 Description

Facial recognition in real-world environments presents significant challenges due to variability in lighting, positioning, and occlusions. This system addresses these issues through a hybrid pipeline that combines the power of Convolutional Neural Networks (CNNs) with classical statistical methods for dimensionality reduction.

The model achieves performance metrics (Precision, Recall, and F1-Score) of **99.8%**, validating its robustness for critical security applications.

## 🚀 Methodology & Architecture

The system workflow is divided into the following stages:

### 1. Feature Extraction (Deep Learning)
We use the pre-trained **VGG-16** model as a Feature Extractor. We leverage the deep layers of the network to obtain rich, abstract representations of faces, maximizing discrimination capability via **Transfer Learning**.

### 2. Dimensionality Reduction (IoT Optimization)
To make execution viable on IoT devices (such as Raspberry Pi), we implement techniques to compress the feature vector without losing relevant information:
* **PCA (Principal Component Analysis):** Reduces non-essential variance.
* **LDA (Linear Discriminant Analysis):** Maximizes separation between classes (different identities).

### 3. Classification & Verification
The final access decision is made by comparing the optimized features using:
* **Distance Metrics:** Cosine, Mahalanobis, and Euclidean distance.
* **SVM (Support Vector Machines):** Used for the final binary classification (Access Granted / Denied).

## ⚡ Focus on IoT and Edge Computing

Unlike traditional implementations that require expensive hardware (GPUs), this project prioritizes **lightweight algebraic calculation**.

By replacing heavy full-model inference with matrix operations on reduced feature vectors (via PCA/LDA), we achieve an agile system capable of running in real-time on modest hardware.

## 📊 Results

The system was evaluated obtaining the following results in the best test cases:

| Metric | Result |
| :--- | :--- |
| **Precision** | 99.8% |
| **Recall** | 99.8% |
| **F1-Score** | 99.8% |

## 🛠 Technologies Used

* **Python 3.x**
* **TensorFlow / Keras** (Base VGG-16 model)
* **Scikit-Learn** (PCA, LDA, SVM)
* **NumPy** (Optimized matrix calculation)
* **OpenCV** (Image pre-processing)

## 📦 Installation & Usage

1. Clone the repository:
   ```bash
   git clone [https://github.com/your-username/facial-recognition-iot.git](https://github.com/your-username/facial-recognition-iot.git)
