# Feature Engineering & Modeling Log

## Dataset source and competition rules

The dataset appears to be a re-skinned version of Kaggle's **Playground Series S4E1: Bank Churn**. The structure is very similar:

* 3 countries: France, Germany, Spain
* `prod_count` ranges from 1–4
* `acc_balance` has a bimodal distribution, with a large spike around 0

The original competition dataset was generated using a deep-learning model trained on the older "Churn Modelling" dataset.

One interesting technique used by some Kaggle participants was using `last_name` to predict nationality through an external surname-nationality dataset. However, this cannot be used here because the competition rules prohibit external data.

Therefore, `last_name` should only be used if useful information can be extracted entirely from the provided dataset, such as:

* Frequency encoding
* Out-of-fold target encoding

No external surname or nationality lookup will be used.

---

# Main finding: Feature engineering has limited impact

A Kaggle discussion tested more than **300 combinations** of:

* Feature sets
* Categorical encodings
* Scaling methods
* Machine-learning models

The main conclusion was that adding large numbers of engineered features produced only **small improvements**.

Model choice and categorical encoding appeared to matter more than creating complicated new features.

The best combination reported in that experiment was approximately:

```text
LightGBM
    +
Robust Scaling
    +
GLMM / target-style categorical encoding
```

This suggests that feature engineering should be kept relatively focused instead of creating hundreds of potentially useless features.

Feature engineering is still worth testing, especially because small F1 improvements can matter in a competition, but model selection and encoding should receive more attention.

---

# Feature engineering experiments

## 1. Zero-balance indicator

`acc_balance` has a strong spike at exactly zero.

A binary feature can therefore be created:

```python
balance_zero = (acc_balance == 0)
```

The original `acc_balance` can still be kept.

This feature should be particularly useful for Logistic Regression and SVM because these models may not naturally capture the difference between zero and non-zero balances.

---

## 2. Treat `prod_count` as categorical

The churn rate changes dramatically depending on the number of products:

| `prod_count` | Churn rate |
| -------------- | ---------: |
| 1              |      34.5% |
| 2              |       6.0% |
| 3              |      88.5% |
| 4              |      87.5% |

This relationship is clearly **non-linear**.

Treating `prod_count` as a normal numerical variable implies something like:

```text
1 → 2 → 3 → 4
```

with a consistent numerical relationship.

That does not match the actual churn pattern.

It is therefore better to treat `prod_count` as a categorical variable using:

* One-hot encoding
* Target encoding

This should be especially useful for linear models such as Logistic Regression.

---

## 3. `country × gender`

Country already has a strong relationship with churn, and gender can further affect the outcome.

An interaction feature can therefore be created:

```text
country_gender
```

with categories such as:

```text
France_Female
France_Male
Germany_Female
Germany_Male
Spain_Female
Spain_Male
```

This allows linear models to learn different churn behavior for each country-gender combination.

---

## 4. `is_active × prod_count`

Both `is_active` and `prod_count` have strong relationships with churn.

Their combination may provide additional information.

Possible combinations include:

```text
Active + 1 product
Active + 2 products
Active + 3 products
Active + 4 products

Inactive + 1 product
Inactive + 2 products
Inactive + 3 products
Inactive + 4 products
```

This interaction is worth testing because certain combinations may represent particularly high-risk customer groups.

---

## 5. Ratio features

A few simple ratios are worth testing:

```python
balance_salary_ratio = acc_balance / estimated_salary
balance_age_ratio = acc_balance / age
```

These appeared as minor but somewhat consistent contributors in several public notebooks.

They are unlikely to completely transform the model, but they are simple enough to test.

Division-by-zero cases should be handled properly.

---

## 6. Credit-score bands

`credit_score` can be converted into broad categories such as:

```text
Poor
Fair
Good
Excellent
```

This is mainly worth testing with Logistic Regression and SVM.

However, the EDA already showed that `credit_score` has very weak relationships with the other variables and the target.

Therefore, this is a **low-priority feature engineering experiment**.

---

# Categorical encoding

Target encoding is another approach worth testing.

Instead of representing:

```text
France
Germany
Spain
```

with one-hot encoding, each category can be represented using information about its relationship with the target.

For example:

```text
France  → ~17% churn
Germany → ~38% churn
Spain   → ~17% churn
```

The Kaggle experiments suggest that GLMM/target-style encoding can work particularly well with LightGBM.

### Leakage prevention

Target encoding must be done carefully.

The target value of a row should not be used to calculate the encoded value for that same row.

The training data should therefore be split into folds:

```text
Training data
      ↓
   K folds
      ↓
Use other folds to calculate encoding
      ↓
Encode the held-out fold
```

This produces **out-of-fold target encoding**.

For validation and test data, the encoding should be calculated using the training data only.

---

# Identifier columns

`customer_id` and `last_name` should probably be dropped as raw features.

`customer_id` is simply an identifier and is unlikely to contain meaningful information.

`last_name` could potentially contain useful geographic information, but the strongest known approach relies on an external surname-nationality dataset, which is not allowed here.

Therefore, the initial approach will be:

```text
Drop customer_id
Drop last_name
```

Any useful information from `last_name` should only be extracted using the provided dataset.

---

# Important difference: F1 instead of ROC-AUC

The original Kaggle competition used **ROC-AUC** as its evaluation metric.

This competition uses **F1 score**.

That changes the modeling strategy.

ROC-AUC mainly evaluates how well the model ranks positive and negative examples.

F1 depends on the final classification:

```text
0 → No churn
1 → Churn
```

and balances:

* Precision
* Recall

Therefore, using the default probability threshold of `0.5` is not necessarily optimal.

---

# Decision threshold optimization

Instead of automatically doing:

```python
prediction = probability > 0.5
```

different thresholds should be tested.

For example:

| Threshold |             F1 |
| --------: | -------------: |
|      0.20 |           0.56 |
|      0.25 |           0.59 |
|      0.30 |           0.62 |
|      0.35 | **0.64** |
|      0.40 |           0.63 |
|      0.45 |           0.61 |
|      0.50 |           0.58 |

If `0.35` produces the highest validation F1, then `0.35` should be used instead of `0.5`.

This could provide a larger improvement than some of the more complicated feature engineering experiments.

The threshold should be optimized using validation or out-of-fold predictions rather than the final test set.

---

# Class imbalance

The dataset contains roughly:

```text
Non-churn → ~79%
Churn     → ~21%
```

Therefore, the classes are imbalanced.

A model predicting every customer as non-churn would achieve around 79% accuracy while being completely useless for identifying churn.

Since the evaluation metric is F1, class imbalance should be handled explicitly.

Possible approaches:

### Logistic Regression

```python
class_weight="balanced"
```

### SVM

```python
class_weight="balanced"
```

### Random Forest

```python
class_weight="balanced"
```

### XGBoost

Use:

```python
scale_pos_weight
```

based on the ratio between negative and positive samples.

These settings should be treated as experiments rather than assumptions that they will automatically improve F1.

---

# Planned modeling approach

The experiments will be performed in stages.

## Stage 1: Baseline

Start with:

```text
Original useful features
+
Proper missing-value handling
+
Categorical encoding
+
Scaling where required
+
Class balancing
```

This establishes a reliable baseline F1 score.

---

## Stage 2: Add simple engineered features

Test:

```text
balance_zero
country × gender
is_active × prod_count
balance / estimated_salary
balance / age
```

Each addition should be evaluated using validation F1.

Features that do not improve validation performance should not automatically be kept.

---

## Stage 3: Test categorical encodings

Compare:

```text
One-hot encoding
vs
Target encoding
vs
GLMM-style encoding
```

The main focus should be on:

```text
country
prod_count
```

---

## Stage 4: Optimize the F1 threshold

Generate validation/out-of-fold probabilities and search for the threshold that maximizes F1.

The default `0.5` threshold should not be assumed to be optimal.

---

## Stage 5: Compare models

The main models to test are:

```text
Logistic Regression
SVM
Random Forest
XGBoost
LightGBM
CatBoost
```

All models should be compared using the same validation strategy and **F1 score**.

---

# Main conclusions

The most important points from this investigation are:

1. **External surname data cannot be used** because external data is prohibited.
2. **Massive feature engineering does not appear to provide huge gains** on this dataset.
3. **`prod_count` should be treated as categorical** because its relationship with churn is strongly non-linear.
4. **`balance_zero` is worth adding** because of the unusual zero-balance spike.
5. **A few interactions are worth testing**, especially:

   * `country × gender`
   * `is_active × prod_count`
6. **Categorical encoding may matter more than complicated feature engineering.**
7. **Target/GLMM encoding is worth testing**, especially with LightGBM.
8. **The competition uses F1 rather than ROC-AUC**, so the original Kaggle solutions cannot be followed blindly.
9. **The classification threshold should be optimized** instead of automatically using `0.5`.
10. **Class imbalance needs to be considered**, since only about 21% of customers churn.
11. **Data leakage must be avoided**, particularly during target encoding and validation.

The overall strategy is to start with a strong, simple baseline and add features or modeling techniques only when they produce a measurable improvement in held-out F1.

# Sources:

* [Playground Series S4E1 — Bank Churn (competition overview)](https://www.kaggle.com/competitions/playground-series-s4e1)
* [300+ pipeline experiment discussion (LGBM + robust scaling + GLMM encoding best; FE gains marginal)](https://www.kaggle.com/competitions/playground-series-s4e1/discussion/465992)
* [Surname Nationality Classification dataset (the external-data trick — not permitted under your rules)](https://www.kaggle.com/datasets/alenic/surname-dataset-classification)
* [New features and target encoding for the Surname (notebook)](https://www.kaggle.com/code/thierryneusius/new-features-and-target-encoding-for-the-surname)
* [ShabGaming/Bank-Customer-Churn-Binary-Classification (baseline approach, drops identifiers)](https://github.com/ShabGaming/Bank-Customer-Churn-Binary-Classification)
* [XGBoost for the Kaggle Bank Churn Dataset (ordinal encoding, scale_pos_weight)](https://xgboosting.com/xgboost-for-the-kaggle-bank-churn-dataset/)
