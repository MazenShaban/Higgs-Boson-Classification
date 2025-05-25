# Higgs Boson Event Classification with Machine Learning

This project applies machine learning techniques to classify events from the ATLAS experiment at CERN as either **signal** (events likely involving the Higgs boson) or **background** (other standard model processes).

## 🚀 Project Summary

The goal is to build a classification model that can distinguish Higgs boson events from non-Higgs events using physics-derived features, and ultimately deploy this model via a web interface.

## 📁 Repository Contents

- `Higgs Boson.ipynb`: Main Jupyter notebook containing:
  - Data exploration
  - Preprocessing pipeline
  - Feature selection
  - Model training and evaluation (e.g., SVM)
  - Hyperparameter tuning using `RandomizedSearchCV`
  - (Optional) Saving model for deployment

- `svm_model.pkl`: Trained SVM pipeline saved using `joblib`.

## 🧪 Dataset

This project uses the **Higgs Boson Challenge dataset** from the [Kaggle competition](https://www.kaggle.com/competitions/higgs-boson):

- Each row represents an event detected by the particle accelerator.
- Each column is a feature derived from the detector's measurements.
- The target is binary:
  - `signal`: Higgs boson candidate (1)
  - `background`: Non-Higgs events (0)

## ⚙️ Machine Learning Pipeline

- **Preprocessing**
  - Log-transform skewed numerical features
  - Scaling with `RobustScaler`
  - (Optional) SMOTE for class imbalance

- **Model**
  - Support Vector Machine (SVM)
  - Hyperparameter tuning via `RandomizedSearchCV`

- **Evaluation**
  - Accuracy, Precision, Recall, F1-Score
  - Confusion Matrix
  - ROC-AUC Score
