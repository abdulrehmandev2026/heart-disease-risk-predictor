# ❤️ Heart Disease Risk Predictor

A machine learning web application that predicts the risk of heart disease from clinical patient data. Built entirely in Python using **Streamlit** for the frontend and **scikit-learn** for the ML backend — no separate server required.

🔗 **Live Demo:** https://heart-disease-risk-predictor-ywjstkrvzjtkeplbbmu7.streamlit.app

---

## 📂 Dataset

**UCI Heart Disease Dataset (Cleveland)**
- **File used:** `heart.csv`
- **Rows:** 297 clean patient records (303 original, 6 missing values removed)
- **Features:** 13 clinical attributes
- **Target:** `0` = No Heart Disease, `1` = Heart Disease Present

### Feature Descriptions
| Feature | Description | Type |
| --- | --- | --- |
| `age` | Age in years | Numeric |
| `sex` | Sex (1 = Male, 0 = Female) | Binary |
| `cp` | Chest pain type (0–3) | Categorical |
| `trestbps` | Resting blood pressure (mm Hg) | Numeric |
| `chol` | Serum cholesterol (mg/dl) | Numeric |
| `fbs` | Fasting blood sugar > 120 mg/dl (1/0) | Binary |
| `restecg` | Resting ECG results (0–2) | Categorical |
| `thalach` | Maximum heart rate achieved | Numeric |
| `exang` | Exercise-induced angina (1/0) | Binary |
| `oldpeak` | ST depression induced by exercise | Numeric |
| `slope` | Slope of peak exercise ST segment (0–2) | Categorical |
| `ca` | Number of major vessels colored (0–3) | Numeric |
| `thal` | Thalassemia type (0–3) | Categorical |

---

## 📋 Project Description

This application provides an end-to-end heart disease prediction pipeline:

1. **Data Loading** — reads `heart.csv`, removes duplicates and missing values
2. **EDA** — count plot, correlation heatmap, age vs disease boxplot
3. **Model Training** — trains and evaluates 4 ML classifiers
4. **Model Selection** — picks the best model by ROC-AUC score
5. **Interactive Frontend** — enter patient data → instant risk probability + Low/Moderate/High band

---

## 🛠️ Technologies Used

| Layer | Library | Purpose |
| --- | --- | --- |
| Frontend | `streamlit` | UI, layout, sliders, forms |
| Data | `pandas` | Data loading and manipulation |
| Numerics | `numpy` | Array operations |
| ML Backend | `scikit-learn` | Model training, scaling, evaluation |
| Persistence | `joblib` | Saving trained model and scaler |

**ML Models trained and compared:**
- Logistic Regression
- Random Forest Classifier ✅ (Best)
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)

---

## 📊 Model Performance

| Model | Accuracy | Recall | F1 | ROC-AUC |
| --- | --- | --- | --- | --- |
| Logistic Regression | 83.3% | 82.1% | 83.5% | 0.895 |
| **Random Forest** | **86.7%** | **82.1%** | **87.0%** | **0.941** |
| KNN | 88.3% | 82.1% | 86.8% | 0.908 |
| SVM | 85.0% | 78.6% | 81.0% | 0.919 |

Train / Test split: 80% / 20% (stratified). Recall is reported because missing a sick patient is more dangerous than a false alarm.

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.9 or higher
- `pip` package manager

### 1. Clone the project
