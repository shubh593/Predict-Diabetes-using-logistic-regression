# Predict-Diabetes-using-logistic-regression
For a **Diabetes Prediction** project, your README should highlight the transition from medical health metrics to a binary classification (Diabetes vs. No Diabetes). Since you are using **Logistic Regression**, it is essential to explain how the model uses a probability threshold to make a diagnosis.

---

## 🩺 Predict Diabetes using Logistic Regression

### 📋 Overview

This project implements a machine learning model to predict whether a patient has diabetes based on specific diagnostic measurements. Using **Logistic Regression**, the system analyzes features such as glucose levels, blood pressure, and BMI to calculate the probability of the disease being present.

### 🧠 The Logic: Logistic Regression & The Sigmoid Function

Unlike Linear Regression which predicts continuous values, Logistic Regression predicts **probability**. It uses the **Sigmoid Function** to map any real-valued number into a value between 0 and 1.

* **If Probability  0.5**: The model classifies the patient as "Diabetic" (1).
* **If Probability < 0.5**: The model classifies the patient as "Non-Diabetic" (0).

The mathematical formula used is:


---

### 🚀 Key Features

* **Medical Feature Analysis**: Processes 8 key health indicators (Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, Age).
* **Binary Classification**: Clear output indicating the likelihood of diabetes.
* **Feature Scaling**: Implements `StandardScaler` to ensure features like "Insulin" (high range) don't overpower "Diabetes Pedigree Function" (low range).
* **Model Evaluation**: Uses a **Confusion Matrix** to track True Positives (correctly identified diabetes) and False Negatives (missed cases).

---

### 🛠️ Tech Stack

* **Language**: Python 3.x
* **Machine Learning**: `scikit-learn`
* **Data Processing**: `pandas`, `numpy`
* **Visualization**: `seaborn` (for Confusion Matrix and Histograms)
* **Dataset**: [Pima Indians Diabetes Dataset](https://www.kaggle.com/uciml/pima-indians-diabetes-database)

---

### 📈 Project Workflow

1. **Data Preprocessing**: Identifying "zero" values in columns like Blood Pressure or BMI and replacing them with the Mean/Median (as these values cannot physically be zero).
2. **Exploratory Data Analysis (EDA)**: Using boxplots to identify outliers in glucose and insulin levels.
3. **Train-Test Split**: Splitting data to ensure the model can generalize to new patients.
4. **Scaling**: Applying `StandardScaler` so all features have a mean of 0 and a variance of 1.
5. **Model Training**: Fitting the `LogisticRegression` classifier.

---

### 📊 Model Performance

In medical predictions, **Recall** (Sensitivity) is often more important than Accuracy, as we want to minimize the number of diabetic patients who are incorrectly told they are healthy.

| Metric | Score |
| --- | --- |
| **Accuracy** | 0.78 (Example) |
| **Precision** | 0.72 |
| **Recall (Sensitivity)** | 0.75 |

---

### 📦 Quick Start

1. **Install Requirements**:
```bash
pip install sklearn pandas seaborn matplotlib

```


2. **Run Inference**:
```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Load your trained model and scaler
# Example input: [6, 148, 72, 35, 0, 33.6, 0.627, 50]
prediction = model.predict(scaler.transform([new_patient_data]))

if prediction[0] == 1:
    print("High Risk: Potential Diabetes Detected")
else:
    print("Low Risk: No Diabetes Detected")

