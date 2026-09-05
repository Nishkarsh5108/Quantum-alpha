# NaN Handling Strategy Implementation

This document outlines how missing values (NaNs) are handled in the Quantum Alpha dataset, based on the implementation in `src/Manya/data_preprocessing.py`.

The strategy prioritizes using available data to make educated imputations rather than blindly filling with global averages or modes, which could mislead our classification model.

## 1. Product Count (`prod_count`)
- **Missing:** ~5.4%
- **Strategy:** Machine Learning Imputation (Random Forest)
- **Implementation:** `prod_count` is the most important feature for predicting churn. Filling it with the mode (2) is dangerous because customers with 2 products have the lowest churn rate. Instead, we train a `RandomForestClassifier` on the rows where `prod_count` is known. We use features like `credit_score`, `country`, `gender`, `age`, `tenure`, `acc_balance`, `has_card`, `is_active`, and `estimated_salary` to predict the missing values. 
- **Important Note:** We strictly exclude the target variable (`exit_status`) when training this imputer to prevent data leakage.

## 2. Account Balance (`acc_balance`)
- **Missing:** ~8%
- **Strategy:** Grouped Median Imputation
- **Implementation:** The distribution of account balances varies drastically by country (e.g., France and Spain have many zero-balance accounts, whereas Germany does not). We calculate the median `acc_balance` separately for each country. If a row is missing its `acc_balance`, it is filled with the median balance of that specific customer's country. If the country is also unknown, it falls back to the global median.

## 3. Country (`country`)
- **Missing:** ~6.7%
- **Strategy:** Constant Imputation
- **Implementation:** Missing countries are replaced with the string `"Unknown"`. This ensures we don't lose the information that the data was missing, and allows it to be treated as a distinct category during one-hot encoding later.

## 4. Credit Score (`credit_score`)
- **Missing:** ~10.6%
- **Strategy:** Global Median Imputation
- **Implementation:** `credit_score` has very low correlation with other features in the dataset. Therefore, simple median imputation is sufficient and safe. Missing values are filled with the overall median credit score of the training data.

## Execution
This pipeline is encapsulated in the `QuantumAlphaImputer` class in `src/Manya/data_preprocessing.py`. The imputer is strictly `.fit()` on the training data, and then used to `.transform()` both the training and test datasets. This produces `train_clean.csv` and `test_clean.csv`, which have 0 missing values and are ready for model training.
