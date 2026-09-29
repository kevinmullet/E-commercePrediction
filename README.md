# E-commerce Customer Churn Analysis & Prediction

Exploratory data analysis and churn prediction on a synthetic e-commerce customer dataset, including a diagnostic investigation into why the initial models underperformed.

## Dataset

- **Source:** [E-commerce Customer Data / Custom Ratios](https://www.kaggle.com/) — Kaggle (add the exact dataset link here)
- **Size:** 250,000 rows, 13 columns
- **Note:** This dataset is **synthetic**, generated for practice purposes. Findings below should be read as a demonstration of analysis and diagnostic process, not as real business insight.
- **Target column:** `Churn` (0 = active customer, 1 = churned), imbalanced at roughly 80:20 (4:1)

To reproduce this notebook, download the CSV from Kaggle and place it in the same folder before running.

## Workflow

1. **Data cleaning**
   - Filled missing values in `Returns` with 0 (assumed to mean "no return")
   - Dropped identifier columns (`Customer ID`, `Customer Name`) and a duplicate `Age`/`Customer Age` column

2. **Feature extraction**
   - Converted `Purchase Date` to datetime and extracted `Purchase Month` and `Purchase Weekday`

3. **Exploratory Data Analysis**
   - Univariate distributions (Age, Total Purchase Amount, Product Category, etc.)
   - Bivariate analysis: Payment Method, Gender, Age vs. Churn
   - Churn rate found nearly identical across all customer segments (e.g. ~20% for both Male and Female customers)

4. **Encoding**
   - One-hot encoded `Product Category`, `Payment Method`, and `Gender` using `pd.get_dummies(drop_first=True)`

5. **Modeling**
   - Stratified 80/20 train-test split to preserve class balance
   - **Random Forest** (`class_weight='balanced'`)
   - **Logistic Regression** (`class_weight='balanced'`, features scaled with `StandardScaler`)

6. **Diagnosis**
   - Compared built-in feature importance (Mean Decrease Impurity) against **permutation importance** on the test set, since MDI is biased toward continuous features with many split points

## Results

| Model | Churn Recall | Churn Precision | Accuracy |
|---|---|---|---|
| Random Forest | ~0.2% | ~18% | ~80% |
| Logistic Regression | ~50% | ~20% | ~50% |

- Random Forest almost never predicts churn, defaulting to the majority class.
- Logistic Regression predicts churn far more often, but precision stays close to the 20% base rate — indistinguishable from guessing.
- **Permutation importance** for every feature was near zero, with several features slightly negative, meaning none of them meaningfully help predict churn on unseen data.

## Conclusion

Both models converge on the same finding through different failure modes: **the `Churn` label in this dataset has no meaningful relationship with the available features.** This is consistent with how many synthetic datasets are generated, where the target column is sampled independently of the other fields rather than from a real underlying process.

The value of this project is in the diagnostic process — using permutation importance and a second model as a cross-check to correctly attribute the poor performance to the data rather than to model choice or tuning.

## Tools

Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn — run in Google Colab.

## How to run

Open the notebook in Google Colab, upload the dataset CSV, and run all cells in order.
