# Loan Eligibility Prediction using Support Vector Machine (SVM)

This repository contains a comprehensive data preprocessing and machine learning pipeline to predict loan eligibility using a Support Vector Machine (SVM) classifier. 

The pipeline handles raw loan applicant data, addresses missing values, encodes categorical features, and prepares the data for robust model training.

---

## 📌 Project Overview
The objective of this project is to clean, preprocess, and build a predictive model to determine whether a loan applicant is likely to be approved or rejected based on key financial and demographic attributes.

### Dataset Features Analysed:
* **Demographics:** `Gender`, `Married`, `Dependents`, `Education`, `Self_Employed`
* **Financials:** `ApplicantIncome`, `CoapplicantIncome`, `LoanAmount`, `Loan_Amount_Term`

---

## 🛠️ Data Preprocessing Pipeline

### 1. Handling Missing (Null) Values
To ensure no data loss, custom imputation strategies were applied based on feature types:
* **Numerical Features (`LoanAmount`, `Loan_Amount_Term`):** Imputed using the **Median** to prevent outlier distortion.
* **Categorical Features (`Gender`, `Dependents`, `Self_Employed`):** Imputed using the **Mode** (most frequent value).

### 2. Feature Engineering & Encoding
* **Label Encoding:** Converted binary and categorical text features (like `Gender`, `Married`, `Education`, `Self_Employed`) into numerical integers (`0` and `1`) using `LabelEncoder`.
* **Data Cleaning:** Handled the `Dependents` column by replacing `'3+'` with `3` and converting the series to an integer type.
* **Feature Selection:** Dropped irrelevant identifiers such as `Loan_ID` to prevent model overfitting.

### 3. Custom Target Variable Generation
A logical rule-based engine was used to generate the target label (`Loan_status`):
$$\text{Loan\_status} = \begin{cases} 1 & \text{if } \text{ApplicantIncome} > (\text{LoanAmount} \times 15) \\ 0 & \text{otherwise} \end{cases}$$

> ⚠️ **Note on Data Imbalance:** The current threshold generates a highly imbalanced target variable (~95% approved vs ~5% rejected). Future iterations will focus on threshold tuning or applying SMOTE (Synthetic Minority Over-sampling Technique) to balance the classes before model fitting.

---

## 🚀 Upcoming Model Training (SVM)
The next stage of the pipeline involves:
1. **Feature Scaling:** Applying `StandardScaler` to normalize features (critical for distance-based SVM performance).
2. **Train-Test Splitting:** Dividing data with stratified splits to preserve class ratios.
3. **SVM Fitting:** Training an `SVC` (Support Vector Classifier) with `RBF` / `Linear` kernels and evaluating using F1-Score to handle the class imbalance.

---

