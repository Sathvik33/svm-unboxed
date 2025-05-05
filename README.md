# SVM Classification on Breast Cancer Dataset

This project demonstrates how to build and evaluate a **Support Vector Machine (SVM)** classifier for binary classification using the **Breast Cancer Wisconsin (Diagnostic) dataset**. It includes data preprocessing, visualization, model training using different kernels, hyperparameter tuning, and decision boundary plotting using PCA.

---

## 📂 Project Structure

- `data/data.csv` - Dataset file
- `7.py` - Main Python script
- `requirements.txt` - Required dependencies
- `README.md` - Project documentation

---

## 🚀 Features

- Data preprocessing and label encoding
- Feature scaling using `MinMaxScaler`
- Dimensionality reduction with PCA for visualization
- Training SVM models with both:
  - Linear kernel
  - RBF (Gaussian) kernel
- Hyperparameter tuning using `RandomizedSearchCV`
- Decision boundary visualization in 2D space
- Evaluation metrics: Accuracy, Precision, Recall, F1-score

---

## 🏆 Best Model Performance

**Best Hyperparameters from RandomizedSearchCV:**
{'kernel': 'linear', 'gamma': 'scale', 'C': 1.0}



**Best Model Evaluation:**

- **Accuracy:** 0.982
- **Precision:**
  - Class 0 (Benign): 0.97
  - Class 1 (Malignant): 1.00
- **Recall:**
  - Class 0 (Benign): 1.00
  - Class 1 (Malignant): 0.95
- **F1-score:**
  - Class 0 (Benign): 0.99
  - Class 1 (Malignant): 0.98

---

## 📊 Dataset

- **Source**: UCI Machine Learning Repository  
- **Target**: `diagnosis` (0 = Benign, 1 = Malignant)  
- **Features**: 30 numerical attributes related to cell nuclei
