# Predictive Maintenance using Machine Learning

## Description
Predictive maintenance project using classic machine learning techniques to predict equipment failure from multivariate sensor time-series data. The project emphasizes exploratory analysis, rolling feature engineering, and leakage-aware validation rather than deep learning.

---

## Dataset
- **Source:** NASA CMAPSS (FD001)
- **Records:** ~20,000+
- **Engines:** 100
- **Features:** 3 operational settings, 21 sensor readings
- **Target:** Binary failure indicator (failure within next 30 cycles)

---

## Approach

### Exploratory Data Analysis
- Identified and removed near-constant sensors
- Analyzed sensor distributions and degradation trends
- Examined inter-sensor correlations

### Feature Engineering
- Computed Remaining Useful Life (RUL) per engine
- Reframed RUL as a binary failure prediction task
- Engineered rolling mean and standard deviation features per engine
- Removed leakage-prone and non-informative features

### Modeling & Evaluation
- Group-aware train–test split to prevent engine-level data leakage
- Trained:
  - Logistic Regression
  - Random Forest
- Evaluated using ROC–AUC

---

## Results
- **Logistic Regression ROC–AUC:** ~0.99  
- **Random Forest ROC–AUC:** ~0.99  

Both models achieved comparable performance, indicating strong feature separability driven by rolling statistical features.

---

## Tech Stack
- Python
- pandas, NumPy
- scikit-learn
- matplotlib, seaborn

---

## Repository Structure
```text
predictive_maintenance/
│
├── data/
│   └── FD001/
│       ├── train_FD001.txt
│       ├── test_FD001.txt
│       └── RUL_FD001.txt
│
├── notebooks/
│   └── 01_eda.ipynb
│
├── README.md
└── requirements.txt
