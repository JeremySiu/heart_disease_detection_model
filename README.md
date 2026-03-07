# Heart Disease Prediction Model

A machine learning project that leverages clinical data to predict the presence of heart disease using ensemble learning and logistic regression techniques.

## 🎯 Project Overview

Cardiovascular diseases (CVDs) are the leading cause of global mortality, claiming 17.9 million lives annually. This project bridges the gap between static clinical data and dynamic patient care by developing a predictive model that can identify high-risk individuals before a major cardiac event occurs. The model serves as the backend for a full-stack application designed to provide accessible, non-invasive screening tools for heart health assessment.

## 📊 Dataset

The project uses the **Heart Failure Prediction Dataset**, which consolidates five independent heart disease datasets from the UCI Machine Learning Repository, creating a comprehensive collection of **918 observations** with **11 clinical features**.

### Features

| Feature | Description | Type |
|---------|-------------|------|
| **Age** | Age of the patient (years) | Numeric |
| **Sex** | M: Male, F: Female | Categorical |
| **ChestPainType** | TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic | Categorical |
| **RestingBP** | Resting blood pressure (mm Hg) | Numeric |
| **Cholesterol** | Serum cholesterol (mm/dl) | Numeric |
| **FastingBS** | Fasting blood sugar > 120 mg/dl (1: Yes, 0: No) | Binary |
| **RestingECG** | Normal, ST: ST-T wave abnormality, LVH: Left ventricular hypertrophy | Categorical |
| **MaxHR** | Maximum heart rate achieved (60-202) | Numeric |
| **ExerciseAngina** | Exercise-induced angina (Y: Yes, N: No) | Binary |
| **Oldpeak** | ST depression value | Numeric |
| **ST_Slope** | Up: Upsloping, Flat: Flat, Down: Downsloping | Categorical |
| **HeartDisease** | Target variable (1: Heart disease, 0: Normal) | Binary |

## 🔧 Data Preprocessing

### 1. Missing Value Handling
- Replaced 0 values in `RestingBP` (1 instance) with median
- Replaced 0 values in `Cholesterol` (172 instances) with median

### 2. Feature Engineering
- **Binary Conversion**: Mapped `Sex` (M→1, F→0) and `ExerciseAngina` (Y→1, N→0)
- **One-Hot Encoding**: Applied to `ChestPainType`, `RestingECG`, and `ST_Slope` to create independent feature columns

### 3. Correlation Analysis
Key findings from Pearson correlation matrix:
- **ST_Slope_Flat** (0.6): Strongest positive predictor
- **ExerciseAngina** (0.5): Significant indicator
- **ST_Slope_Up** (-0.6): Strong negative correlation (healthy indicator)
- **MaxHR** (-0.4): Lower max heart rate linked to disease

## 🤖 Models

### Logistic Regression
Uses Maximum Likelihood Estimation (MLE) to establish a decision boundary through a sigmoid function, outputting probabilities between 0 and 1.

### Random Forest Classifier ⭐ (Selected Model)
An ensemble learning method using Bootstrap Aggregating (Bagging) with `max_depth=5`. Constructs multiple decision trees and outputs the majority vote, capturing non-linear interactions between features.

## 📈 Model Performance

### Logistic Regression Results
- **Accuracy**: 86.4%
- **Balanced Accuracy**: 86.2%
- **Precision**: 88.6%
- **Recall**: 86.1%
- **F2-Score**: 86.2%
- **AUC**: 0.929

### Random Forest Results (Winner) 🏆
- **Accuracy**: 88.0%
- **Balanced Accuracy**: 87.8%
- **Precision**: 89.4%
- **Recall**: 88.1%
- **F2-Score**: 88.0%
- **AUC**: 0.946

### Why Random Forest?
The Random Forest model was selected for deployment due to:
- **Superior recall**: Minimizes false negatives (missed disease cases)
- **Higher AUC**: Better discriminatory power (0.946 vs 0.929)
- **Non-linear relationships**: Captures complex feature interactions
- **Clinical reliability**: Better handles class imbalances

## 🗂️ Project Structure

```
Heart Disease Detection Model/
├── Heart_Disease_Prediction_Model.ipynb  # Main analysis notebook
├── data.csv                              # Dataset
├── model.joblib                          # Saved Random Forest model
├── README.md                             # Project documentation
```

## 🚀 Usage

### Requirements
```python
numpy
pandas
scikit-learn
seaborn
matplotlib
joblib
```

### Loading the Model
```python
import joblib

# Load the trained model
model = joblib.load('model.joblib')

# Make predictions
prediction = model.predict(processed_features)
```

## 🎯 Key Insights

1. **ST_Slope_Flat** is the strongest predictor of heart disease (correlation: 0.6)
2. **Exercise-induced angina** significantly correlates with heart disease presence
3. **Lower maximum heart rate** during stress testing indicates cardiovascular impairment
4. Random Forest outperforms Logistic Regression in all metrics, particularly in minimizing false negatives—critical for medical diagnostics

## 📚 References

fedesoriano. (September 2021). Heart Failure Prediction Dataset. Retrieved March 2026 from https://www.kaggle.com/fedesoriano/heart-failure-prediction

---

**Note**: This model is intended for educational and research purposes. It should not be used as a substitute for professional medical diagnosis or treatment.