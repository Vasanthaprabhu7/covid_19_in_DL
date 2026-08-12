# AI-Based COVID-19 Detection Using Chest X-Ray Images

## 📌 Project Overview

**AI-Based COVID-19 Detection Using Chest X-Ray Images** is a deep learning project that uses **Convolutional Neural Networks (CNNs)** to classify chest X-ray images into two categories:

* 🦠 **COVID-19**
* 🫁 **Normal**

The main goal is to develop an AI-based system that can analyze chest X-ray images and provide rapid classification support for COVID-19 screening. The project focuses on image preprocessing, CNN model development, model training, and performance evaluation.

> **Note:** This project is intended for educational and research purposes and should not be considered a replacement for professional medical diagnosis.

---

## 🎯 Objectives

* Detect COVID-19 patterns from chest X-ray images.
* Provide quick AI-based screening support.
* Apply image processing and machine learning techniques to medical images.
* Build and compare CNN architectures.
* Evaluate models using accuracy and COVID-19 recall.
* Study the effect of model complexity on generalization and overfitting.

---

## 📊 Dataset

The notebook uses a dataset containing two classes:

| Class    | Description                                      |
| -------- | ------------------------------------------------ |
| COVID-19 | Chest X-ray image associated with COVID-19       |
| Normal   | Chest X-ray image from a normal/healthy category |

### Dataset Files

```text
CovidImages.npy
CovidLabels.csv
```

### Dataset Dimensions

```text
Total Images : 251
Image Size   : 128 × 128
Channels     : 3 (RGB)
```

The images are stored as NumPy arrays, while the corresponding labels are stored in a CSV file.

---

## 🛠️ Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* OpenCV
* TensorFlow
* Keras
* Scikit-learn
* Google Colab

---

## 📚 Libraries

The project uses libraries for:

* Data analysis — `NumPy`, `Pandas`
* Visualization — `Matplotlib`, `Seaborn`
* Image processing — `OpenCV`
* Deep learning — `TensorFlow`, `Keras`
* Data splitting — `train_test_split`
* Label encoding — `LabelEncoder`
* Evaluation — `classification_report`, `confusion_matrix`

---

## 🔄 Project Workflow

```text
Chest X-Ray Dataset
        ↓
Load Images & Labels
        ↓
Data Exploration
        ↓
Image Preprocessing
        ↓
Train / Validation Split
        ↓
CNN Model 1
        ↓
Model Evaluation
        ↓
CNN Model 2
        ↓
Model Evaluation
        ↓
Compare Model Performance
```

---

## 🔧 Data Preprocessing

The notebook performs preprocessing before training the neural networks.

The workflow includes:

1. Loading the `.npy` image data.
2. Loading labels from the CSV file.
3. Exploring image dimensions and sample images.
4. Preparing the image data for CNN training.
5. Normalizing image values.
6. Splitting the data into training and validation sets.
7. Preparing labels for binary classification.

A random seed of **42** is used to improve reproducibility during random operations and model initialization.

---

# 🧠 CNN Model 1

The first CNN contains **three convolutional and pooling combinations**, followed by a fully connected layer and a sigmoid output layer.

### Architecture

```text
Input: 128 × 128 × 3
        ↓
Conv2D - 32 Filters
        ↓
MaxPooling2D
        ↓
Conv2D - 64 Filters
        ↓
MaxPooling2D
        ↓
Conv2D - 128 Filters
        ↓
MaxPooling2D
        ↓
Flatten
        ↓
Dense - 128 neurons
        ↓
Sigmoid Output
```

The model uses:

* ReLU activation in convolutional and hidden layers
* Sigmoid activation for binary classification
* Binary cross-entropy loss
* Adam optimizer
* Accuracy as the evaluation metric

## The first model contains **4,287,809 trainable parameters**.

# 🧠 CNN Model 2

The second CNN was introduced to reduce model complexity and improve generalization.

It uses **two convolutional and pooling combinations** instead of three.

### Architecture

```text
Input: 128 × 128 × 3
        ↓
Conv2D - 32 Filters
        ↓
MaxPooling2D
        ↓
Conv2D - 64 Filters
        ↓
MaxPooling2D
        ↓
Flatten
        ↓
Dense - 128
        ↓
Dense - 64
        ↓
Sigmoid Output
```

The model uses:

```text
Optimizer  : Adam
Learning Rate : 0.0001
Loss       : Binary Cross-Entropy
Metric     : Accuracy
```

## The second model contains **8,416,449 trainable parameters** according to the notebook's model summary.

## 🏋️ Model Training

The models are trained using the prepared training data.

### CNN Model 1

```text
Epochs     : 10
Batch Size : 8
Training Time : 92.88 seconds
```

The training accuracy increased from approximately **83.43% in Epoch 1** to **100% by Epoch 10**.

### CNN Model 2

```text
Epochs     : 10
Batch Size : 8
Training Time : 87.32 seconds
Learning Rate : 0.0001
```

## The second model reached approximately **98.86% training accuracy** in the final epoch.

## 📈 Model Performance

The notebook compares the CNN models using:

* Training Accuracy
* Validation Accuracy
* COVID-19 Training Recall
* COVID-19 Validation Recall

### Results

| Model                     | Train Accuracy | Validation Accuracy | COVID Recall (Train) | COVID Recall (Validation) |
| ------------------------- | -------------: | ------------------: | -------------------: | ------------------------: |
| CNN Model – 3 Conv Layers |        100.00% |             100.00% |              100.00% |                   100.00% |
| CNN Model – 2 Conv Layers |         99.43% |              97.37% |               98.70% |                   100.00% |

## These are the values recorded in the notebook's final evaluation table.

## 🔍 Evaluation Metrics

### Accuracy

Measures the overall percentage of correctly classified images.

```text
Accuracy = Correct Predictions / Total Predictions
```

### Recall

Recall is particularly important for COVID-19 screening because it measures how effectively the model identifies actual COVID-19 cases.

The notebook calculates COVID-19 recall using the class labeled **`1`**.

---

## 💡 Key Observations

* The first CNN achieved very high training performance.
* The second CNN achieved **97.37% validation accuracy** and **100% COVID-19 validation recall** in the final comparison.
* The notebook highlights the possibility of overfitting when training performance is substantially stronger than validation performance.
* CNN architecture and learning-rate adjustments were explored to improve generalization.

---

## 📁 Project Structure

```text
COVID-19-PREDICTION/
│
├── AI_Based_COVID_19_Detection_Using_Chest_X_Ray_Images.ipynb
│   ├── CovidImages.npy
│   └── CovidLabels.csv
│
└── README.md
```

---

## ▶️ How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/Vasanthaprabhu7/covid_19_in_DL.git
cd AI-Based-COVID-19-Detection
```

### 2. Install Dependencies

```bash
pip install numpy pandas matplotlib seaborn opencv-python tensorflow scikit-learn
```

### 3. Open the Notebook

You can run the notebook using:

* Google Colab
* Jupyter Notebook
* JupyterLab

### 4. Configure Dataset Paths

Update the dataset paths in the notebook according to your local or Google Drive location.

The original notebook loads the dataset from Google Drive.

### 5. Run All Cells

Execute the notebook sequentially to:

```text
Load Dataset
    ↓
Explore Data
    ↓
Preprocess Images
    ↓
Train CNN Models
    ↓
Evaluate Models
    ↓
Compare Results
```

---

## 🚀 Future Enhancements

* Increase the size and diversity of the dataset.
* Apply data augmentation.
* Add dropout and regularization techniques.
* Experiment with transfer learning models such as ResNet, VGG, or EfficientNet.
* Add ROC-AUC and F1-score evaluation.
* Develop a web interface for uploading X-ray images.
* Add explainable AI techniques such as Grad-CAM.
* Validate the model on independent external datasets.

---

## ⚠️ Limitations

The dataset used in this notebook contains only **251 images**, which is relatively small for training a deep learning model. Therefore, the reported high accuracy should not automatically be interpreted as clinical performance.

The model should be further validated using larger, diverse, and independent medical datasets before any real-world clinical application.

---

## 👨‍💻 Project Summary

This project demonstrates how **Artificial Intelligence and Convolutional Neural Networks** can be applied to chest X-ray image classification for COVID-19 screening. The notebook covers the complete workflow from dataset loading and preprocessing to CNN development, training, evaluation, and model comparison.

### Main Concepts

```text
Artificial Intelligence
        ↓
Deep Learning
        ↓
Convolutional Neural Network
        ↓
Medical Image Classification
        ↓
COVID-19 Detection
```

---
