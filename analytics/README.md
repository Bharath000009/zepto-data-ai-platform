
## Outputs

- `titanic.csv` — raw Titanic data (offline fallback, committed)
- `titanic_clean.csv` — cleaned data used by Part B
- `best_pipeline.joblib` — full fitted pipeline (preprocessing + estimator)
- `plots/` — all generated charts (PNG)

---

## Part A — Profiling, Cleaning, EDA

### 1. Loading

The raw dataset was loaded **once** via `sns.load_dataset("titanic")` in
`01_eda.py` and immediately saved to `titanic.csv`. Part B loads the cleaned
CSV — `sns.load_dataset` is never called again anywhere in the module.

Shape: **891 rows × 15 columns**.

### 2. Missing Values

Measured percentages (before any handling):

| Column | Missing % | Strategy |
|--------|-----------|----------|
| `deck` | 77.22% | **>30% → drop column** |
| `age` | 19.87% | **5–30% → median impute** |
| `embarked` | 0.22% | **<5% → drop rows** |
| `embark_town` | 0.22% | **<5% → drop rows** |

Justification:
- `deck` is missing for more than three quarters of passengers. Imputing
  three-quarters of a column would fabricate data; encoding "missing" would
  make the column nearly constant. Dropping is the honest choice.
- `age` sits in the 5–30% band. Median is a robust choice — age is roughly
  symmetric enough that median doesn't distort the distribution.
- `embarked` / `embark_town` are missing in only 2 rows (0.22%).
  Dropping 2 rows out of 891 is negligible.

After cleaning: **889 rows × 14 columns**.

### 3. Univariate Analysis

**IQR-based outlier counts** (points outside `[Q1 − 1.5×IQR, Q3 + 1.5×IQR]`):

| Column | Q1 | Q3 | IQR | Lower | Upper | Outliers |
|--------|----|----|-----|-------|-------|----------|
| `age` | 22.00 | 35.00 | 13.00 | 2.50 | 54.50 | **65** |
| `fare` | 7.90 | 31.00 | 23.10 | −26.76 | 65.66 | **114** |

**Fare skewness:**
- mean = **32.097**
- median = **14.454**
- mode = **8.050**

Ordering mean > median > mode → **right-skewed**. A small number of very
high fares (up to 512 GBP) pull the mean well above the median, and the mode
(most common fare) sits even lower.

### 4. Bivariate Analysis — Survival Rates

**(a) By sex:**
- female: **0.740**
- male:   **0.189**

**(b) By pclass:**
- 1st class: **0.626**
- 2nd class: **0.473**
- 3rd class: **0.242**

**(c) By sex + pclass:**
- female, 1st: **0.967**
- female, 2nd: **0.921**
- female, 3rd: **0.500**
- male,   1st: **0.369**
- male,   2nd: **0.157**
- male,   3rd: **0.135**

The joint breakdown shows that the male/female gap is far larger than the
class gap: 1st-class men (0.369) still survived less than 3rd-class women
(0.500). Sex was the dominant factor.

### 5. Correlation Matrix (6 specified columns)

Computed on exactly `survived, pclass, age, sibsp, parch, fare`.
`adult_male` and `alone` were excluded — they are derived/redundant flags
(directly computable from `sex`/`age` and from `sibsp`+`parch`), not
independent measured features.

**Top 2 strongest correlations by absolute value:**

1. **`fare ↔ pclass`: r = −0.548** — higher class number (lower socioeconomic
   class) → lower fare. Strong, negative, expected.
2. **`parch ↔ sibsp`: r = +0.415** — passengers traveling with siblings/spouses
   tend to also travel with parents/children. Both capture "family on board,"
   hence the positive link.

### 6. Multivariate Charts (4 total)

1. **Survival by class and sex** — group bar chart. Women survive at much
   higher rates than men in every class; the gap widens in 3rd class.
2. **Age by survival status** — boxplot. Surviving passengers skew slightly
   younger; the medians differ by a few years.
3. **Fare vs age by survival** — scatterplot. Survivors cluster at higher
   fares, especially the very high-fare outliers who almost all survived.
4. **Survival by embarkation port and class** — grouped bars. Cherbourg (C)
   has the highest survival rate, partly because more 1st-class passengers
   boarded there.

### 7. Exploratory Standardization (age, fare)

Applied the z-score `z = (x − mean) / std` on the full cleaned DataFrame.

| | age | fare |
|---|---|---|
| Before mean | 29.315 | 32.097 |
| Before std  | 12.985 | 49.698 |
| After mean  | ~0 | ~0 |
| After std   | 1.0006 | 1.0006 |

Confirms the transformation produces columns with mean ≈ 0 and std ≈ 1.
This is **EDA-only**; the modeling pipeline fits its own scaler on the
training split.

---

## Part B — Predictive Modeling

### 8. Stratified Split

Original class balance: **61.75% died, 38.25% survived**.

Stratification matters because the classes are imbalanced (roughly 3:2).
A plain random split could easily put slightly more or fewer survivors in
the test set, biasing every downstream metric. Stratifying forces both the
train and test sets to preserve the same 61.75 / 38.25 split — the observed
train and test balances after splitting confirm this worked:

- Train: {0: 0.6174, 1: 0.3826}
- Test:  {0: 0.6180, 1: 0.3820}

Split sizes: 711 train, 178 test.

### 9. Preprocessing

All preprocessing steps — imputation, one-hot encoding, and scaling — are
wrapped in a `ColumnTransformer` inside a `Pipeline`. The pipeline is
**fit only on `X_train`**, then applied to `X_test` via `.transform()`
(structurally enforced by scikit-learn). No step ever sees the test set
during fitting, so there is zero leakage.

- Numeric (`age, sibsp, parch, fare`): median impute → `StandardScaler`
- Categorical (`sex, embarked`): most-frequent impute → one-hot encode

### 10. Three Classifiers (same train/test split)

| Model | Accuracy | Precision | Recall | F1 | AUC |
|-------|----------|-----------|--------|-----|-----|
| Logistic Regression | 0.7809 | 0.7544 | 0.6324 | 0.6880 | **0.8265** |
| **Decision Tree** | **0.8146** | **0.7869** | **0.7059** | **0.7442** | 0.8207 |
| Random Forest | 0.7978 | 0.7759 | 0.6618 | 0.7143 | 0.8211 |

Confusion matrices (rows = actual, cols = predicted):

- Logistic Regression: `[[96, 14], [25, 43]]`
- Decision Tree:       `[[97, 13], [20, 48]]`
- Random Forest:       `[[97, 13], [23, 45]]`

The Decision Tree was rendered with `plot_tree`, labeled with feature names
and class names, and saved to `plots/decision_tree.png`.

### 11. Imbalance Handling Comparison

For Logistic Regression (same train/test split), three variants:

| Variant | Precision | Recall | F1 |
|---------|-----------|--------|-----|
| Baseline (no handling) | 0.7544 | 0.6324 | 0.6880 |
| `class_weight='balanced'` | 0.7500 | **0.7059** | **0.7273** |
| SMOTE (applied to training fold only) | 0.7460 | 0.6912 | 0.7176 |

**Conclusion:** `class_weight='balanced'` gave the best F1 (0.7273), followed
closely by SMOTE (0.7176). Both meaningfully improved recall over the baseline
(0.6324 → 0.7059 and 0.6912) at a small precision cost. `class_weight` is
slightly preferable here because it is simpler (no synthetic data generation)
and gives the strongest recall on the minority class without any risk of
SMOTE-generated noise.

### 12. Hyperparameter Tuning (Random Forest)

`GridSearchCV` over `n_estimators × max_depth × max_features`:

- **Best parameters:** `n_estimators=50, max_depth=5, max_features='sqrt'`
- **Best CV F1:** 0.7425
- **OOB score:** **0.8073** (out-of-bag accuracy, calculated with
  `oob_score=True` at construction time)

The OOB score (0.8073) is close to the CV F1-implied accuracy, confirming
the tuned forest generalizes reasonably.

### 13. Regression Side-Task — Predicting Fare

Multivariate linear regression with all other available features as inputs.

| Metric | Value |
|--------|-------|
| MAE | **21.0986** |
| RMSE | **41.7021** |
| R² | **0.3482** |
| Adjusted R² | **0.2965** |

Residual plot saved to `plots/residual_plot.png`.

**Heteroscedasticity conclusion:** The residual plot shows a clear fanning
pattern — residuals are tight at low predicted fares and spread out
dramatically at higher predicted fares. This is **heteroscedasticity**
(non-constant variance). It's expected: fares have a heavy right tail, and
the linear model struggles to predict the few very high fares, so those
produce large residuals. A linear regression on this target is technically
misspecified; a log-transform of `fare` would likely help.

### 14. Model Comparison & Recommendation

**Classification metrics** (from Section 10):

| Model | Accuracy | Precision | Recall | F1 | AUC |
|-------|----------|-----------|--------|-----|-----|
| Logistic Regression | 0.7809 | 0.7544 | 0.6324 | 0.6880 | 0.8265 |
| Decision Tree | 0.8146 | 0.7869 | 0.7059 | 0.7442 | 0.8207 |
| Random Forest | 0.7978 | 0.7759 | 0.6618 | 0.7143 | 0.8211 |

**Regression metrics** (separate scale, not comparable to the above):

| Model | MAE | RMSE | R² | Adjusted R² |
|-------|-----|------|-----|-------------|
| Linear Regression (fare) | 21.0986 | 41.7021 | 0.3482 | 0.2965 |

These two metric groups are on completely different scales — classification
metrics are bounded [0,1] accuracy-style scores; regression metrics are in
GBP units (MAE, RMSE) and variance-explained units (R², Adj R²). They are
reported side by side as two distinct blocks, not merged.

**Final recommendation — deploy the Decision Tree.**

It has the best F1 (0.7442) and best accuracy (0.8146) of the three, with
precision (0.7869) and recall (0.7059) both strong. Logistic Regression has
a marginally higher AUC (0.8265 vs 0.8207), but the gap is trivial, and the
Decision Tree's interpretability (the tree diagram is fully readable) is a
real advantage for a survival-style prediction task where stakeholders want
to see the decision rules. Random Forest is the weakest of the three on F1
here — likely because the un-tuned forest overfits the training data slightly
more than the depth-limited tree.

### 15. Saved Artifact

The full fitted pipeline (preprocessing + final estimator together) is saved
to `analytics/best_pipeline.joblib` using `joblib.dump`.

The saved object is a `sklearn.Pipeline` containing:
1. The `ColumnTransformer` (imputer + encoder + scaler)
2. The Decision Tree classifier

Reload test (Section 11 of `02_modeling.py`): the reloaded pipeline predicts
correctly on raw, unpreprocessed `X_test` rows — confirming the artifact is
usable end-to-end on new data without any external preprocessing.