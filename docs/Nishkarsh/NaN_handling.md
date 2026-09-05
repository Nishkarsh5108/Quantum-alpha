
## What the data tells us

This is the usual bank-churn dataset with customers from France, Germany, and Spain. I checked whether the fact that a value is missing tells us anything about whether a customer will leave.

### 1. Missing values themselves don't seem important

The churn rate is almost the same whether a value is missing or not.

For example:

* `prod_count` missing → ~21.7% churn
* `prod_count` present → ~21.1% churn

The missing-value indicators also have almost no relationship with other important variables.

So, **we probably don't need extra "was this value missing?" features**. The important information is the actual value, not whether it was missing.

### 2. But the columns are related to each other

This is important because we can use other columns to make better guesses about missing values.

For example:

* **Country:** Germany has a much higher churn rate (~38%) than France/Spain (~17%).
* **Account balance:** France and Spain have a huge number of customers with exactly zero balance, while Germany almost never has a zero balance.
* **Product count:** This is extremely important for predicting churn.

`prod_count` has these approximate churn rates:

| Products | Churn rate |
| -------- | ---------: |
| 1        |      34.5% |
| 2        |       6.0% |
| 3        |      88.5% |
| 4        |      87.5% |

So getting `prod_count` wrong when filling in a missing value can seriously hurt the model.

---

# What we should do with each column

This section is mainly for **Logistic Regression and SVM**, because they cannot work with NaNs directly.

Random Forest and XGBoost are slightly different, so they are discussed separately below.

## `prod_count`

**Missing: ~5.4%**

This is the most important column, so we should be careful here.

### Don't use the mode

The most common value is probably `2`.

But `2` also has the **lowest churn rate**.

If we replace every missing value with `2`, we're basically telling the model:

> "Whenever we don't know how many products this customer has, assume they have 2."

That would incorrectly make many customers look like low-risk customers.

### Better approach

Use the other customer information to **predict the missing `prod_count`**.

We can train a small classifier using rows where `prod_count` is already known.

Useful features include:

* `age`
* `is_active`
* `acc_balance`
* `country`
* `credit_score`
* `has_card`
* `estimated_salary`
* `tenure`

**Do NOT use `exit_status`** when predicting `prod_count`.

Otherwise, we're leaking the answer into the input. That's basically giving the model the exam's answer key and then bragging about its accuracy.

A small model such as KNN, Logistic Regression, or Random Forest can predict the missing product counts.

Importantly, this classifier must be trained **only on the training data**, then used to fill the validation/test data.

---

## `acc_balance`

**Missing: ~8%**

Don't use the overall median.

The balance distribution is very different between countries.

France and Spain have lots of customers with a balance of exactly `0`, while Germany mostly has non-zero balances.

So instead, calculate the median **separately for each country**:

```text
France  → France median
Spain   → Spain median
Germany → Germany median
```

This preserves the differences between countries.

For the tiny number of rows where **both `country` and `acc_balance` are missing**, use the overall median as a fallback.

---

## `country`

**Missing: ~6.7%**

The easiest and safest approach is to create a separate category:

```text
France
Germany
Spain
Unknown
```

Then one-hot encode it.

This allows the model to learn whether customers whose country is unknown behave differently.

Since missingness itself doesn't seem to contain much information, this probably won't add a huge amount of predictive power, but it's a safe approach.

A more complicated option would be to predict the missing country using other features, especially `acc_balance`.

However, **I'd start with `Unknown`** and only try predictive country imputation if you need extra performance.

---

## `credit_score`

**Missing: ~10.6%**

This one is much less interesting.

`credit_score` has almost no relationship with the other columns or the target. Its correlations are all around `≤ 0.03`.

So there's not much useful information available to predict the missing values.

Just use **median imputation**:

```python
SimpleImputer(strategy="median")
```

That's perfectly reasonable here.

---

# What about tree-based models?

There's an important correction here.

Not every tree-based model handles NaNs automatically.

### Models that can handle NaNs natively

* XGBoost
* LightGBM
* CatBoost
* sklearn `HistGradientBoostingClassifier`

These can receive NaNs directly.

### Random Forest

sklearn's regular `RandomForestClassifier` needs the missing values handled first, so you should **impute the NaNs before giving the data to it**.

Therefore, a practical setup would be:

```text
                    ┌── Logistic Regression
                    │
Imputed data ────────┼── SVM
                    │
                    └── Random Forest

Raw data with NaNs ─── XGBoost
```

For XGBoost, it's worth testing both:

1. **Raw data with NaNs**
2. **Data after imputation**

Even though XGBoost handles NaNs natively, explicit imputation can sometimes still improve performance depending on the dataset.

---

# Final plan

| Column           | Missing | Recommended approach                            |
| ---------------- | ------: | ----------------------------------------------- |
| `prod_count`   |   ~5.4% | Predict missing values using a small classifier |
| `acc_balance`  |     ~8% | Median grouped by`country`                    |
| `country`      |   ~6.7% | Add`"Unknown"` category                       |
| `credit_score` |  ~10.6% | Median imputation                               |
| Other columns    |      — | Handle according to their type/distribution     |

And **all imputation must be fitted only on the training data**.

The basic principle is:

> **Don't blindly replace missing values with the most common/average value when other columns can help you make a better estimate.**

For this particular dataset, `prod_count` deserves the most attention because it has an unusually strong relationship with churn.
