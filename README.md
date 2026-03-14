# Bank Customer Churn Prediction

## Overview

This project predicts whether a **bank customer will leave the bank (churn) or stay** using Machine Learning.
The goal is to help banks identify customers who are likely to churn.

## Dataset

The dataset contains customer information such as:

* Credit Score
* Geography
* Gender
* Age
* Tenure
* Balance
* Number of Products
* Estimated Salary

**Target Variable:**
`Exited` → 1 = Churn, 0 = Not Churn

## Project Workflow

1. **Load Dataset** using Pandas
2. **Data Cleaning** (remove unnecessary columns)
3. **Encoding** categorical variables
4. **Feature Scaling** using `StandardScaler`
5. **Handle Imbalanced Data** using `SMOTE`
6. **Train Model** using `RandomForestClassifier`
7. **Model Evaluation** using Accuracy & Classification Report
8. **Cross Validation** (5-Fold) to validate model performance

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* SMOTE (Imbalanced-learn)

## Result

The model predicts whether a **customer is likely to churn**, helping banks take preventive actions to retain customers.
