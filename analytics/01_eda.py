"""
Zepto Analytics - Part A: EDA
Loads Titanic ONCE via seaborn, profiles it, applies the missing-value
threshold rule, saves titanic.csv (offline fallback) and titanic_clean.csv,
then produces univariate + bivariate + multivariate analysis and a
standardization sanity check.
"""

from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Agg")  # headless rendering so plots save without a display

OUT_DIR = "analytics"
PLOT_DIR = os.path.join(OUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# ============ 1. LOAD ONCE (offline fallback saved immediately) ============
df = sns.load_dataset("titanic")
df.to_csv(os.path.join(OUT_DIR, "titanic.csv"), index=False)

print("=" * 70)
print("1. PROFILING")
print("=" * 70)
print("Shape:", df.shape)
print("\n--- df.info() ---")
df.info()
print("\n--- df.describe() ---")
print(df.describe())

# ============ 2. MISSING VALUES ============
print("\n" + "=" * 70)
print("2. MISSING VALUES (percentage per column)")
print("=" * 70)
missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
missing_pct = missing_pct[missing_pct > 0]
print(missing_pct)

# Threshold rule:
#   <5%   -> drop rows
#   5-30% -> impute
#   >30%  -> drop column or encode as its own category
df_clean = df.copy()

# deck: ~77% missing -> drop column
if "deck" in df_clean.columns:
    df_clean = df_clean.drop(columns=["deck"])
    print("\nStrategy for 'deck': >30% missing -> DROP COLUMN")

# age: ~20% missing -> impute with median
df_clean["age"] = df_clean["age"].fillna(df_clean["age"].median())
print("Strategy for 'age': 5-30% missing -> IMPUTE with median")

# embarked / embark_town: ~0.2% missing -> drop rows
before = len(df_clean)
df_clean = df_clean.dropna(subset=["embarked", "embark_town"])
print(
    f"Strategy for 'embarked' and 'embark_town': <5% missing -> DROP {before - len(df_clean)} rows")

df_clean.to_csv(os.path.join(OUT_DIR, "titanic_clean.csv"), index=False)

# ============ 3. UNIVARIATE ============
print("\n" + "=" * 70)
print("3. UNIVARIATE ANALYSIS (age, fare)")
print("=" * 70)
for col in ["age", "fare"]:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(df_clean[col], bins=30, edgecolor="black")
    axes[0].set_title(f"{col} - histogram")
    axes[0].set_xlabel(col)
    axes[0].set_ylabel("count")
    axes[1].boxplot(df_clean[col])
    axes[1].set_title(f"{col} - boxplot")
    axes[1].set_ylabel(col)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{col}_univariate.png"), dpi=100)
    plt.close()

    q1, q3 = df_clean[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_out = ((df_clean[col] < lower) | (df_clean[col] > upper)).sum()
    print(f"\n{col}: Q1={q1:.2f}, Q3={q3:.2f}, IQR={iqr:.2f}")
    print(f"  Bounds: [{lower:.2f}, {upper:.2f}]")
    print(f"  IQR outliers: {n_out}")

fare_mean = df_clean["fare"].mean()
fare_median = df_clean["fare"].median()
fare_mode = df_clean["fare"].mode()[0]
print(f"\nfare stats:")
print(f"  mean   = {fare_mean:.3f}")
print(f"  median = {fare_median:.3f}")
print(f"  mode   = {fare_mode:.3f}")
if fare_mean > fare_median > fare_mode:
    print("  Conclusion: RIGHT-SKEWED (mean > median > mode)")
elif fare_mean < fare_median < fare_mode:
    print("  Conclusion: LEFT-SKEWED (mean < median < mode)")
else:
    print("  Conclusion: approximately SYMMETRIC")

# ============ 4. BIVARIATE ============
print("\n" + "=" * 70)
print("4. BIVARIATE - SURVIVAL RATES")
print("=" * 70)
print("\n(a) Survival by sex:")
print(df_clean.groupby("sex")["survived"].mean())
print("\n(b) Survival by pclass:")
print(df_clean.groupby("pclass")["survived"].mean())
print("\n(c) Survival by sex + pclass:")
print(df_clean.groupby(["sex", "pclass"])["survived"].mean())

# ============ 5. CORRELATION MATRIX (exactly 6 columns) ============
print("\n" + "=" * 70)
print("5. CORRELATION MATRIX (6 specified columns)")
print("=" * 70)
corr_cols = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
corr = df_clean[corr_cols].corr()
print(corr)

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
ax.set_title("Correlation matrix (6 specified columns)")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "correlation_heatmap.png"), dpi=100)
plt.close()

# Strongest 2 off-diagonal correlations (by absolute value)
pairs = corr.where(~np.eye(len(corr), dtype=bool)).unstack().dropna()
pairs = pairs[pairs.index.get_level_values(
    0) < pairs.index.get_level_values(1)]
top2 = pairs.abs().sort_values(ascending=False).head(2)
print("\nTop 2 strongest correlations by absolute value:")
for (a, b), _ in top2.items():
    print(f"  {a} <-> {b}: r = {corr.loc[a, b]:+.3f}")

# ============ 6. MULTIVARIATE CHARTS ============
print("\n" + "=" * 70)
print("6. MULTIVARIATE CHARTS")
print("=" * 70)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

sns.barplot(data=df_clean, x="pclass", y="survived", hue="sex", ax=axes[0, 0])
axes[0, 0].set_title("Survival by class and sex")

sns.boxplot(data=df_clean, x="survived", y="age", ax=axes[0, 1])
axes[0, 1].set_title("Age by survival status")

sns.scatterplot(data=df_clean, x="age", y="fare",
                hue="survived", alpha=0.6, ax=axes[1, 0])
axes[1, 0].set_title("Fare vs age by survival")

sns.barplot(data=df_clean, x="embarked",
            y="survived", hue="pclass", ax=axes[1, 1])
axes[1, 1].set_title("Survival by embarkation port and class")

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "multivariate_charts.png"), dpi=100)
plt.close()
print(f"Saved 4 multivariate charts to {PLOT_DIR}/multivariate_charts.png")

# ============ 7. EXPLORATORY STANDARDIZATION ============
print("\n" + "=" * 70)
print("7. EXPLORATORY STANDARDIZATION (age, fare)")
print("=" * 70)
print("Before scaling:")
print(df_clean[["age", "fare"]].agg(["mean", "std"]))

scaler = StandardScaler()
scaled = scaler.fit_transform(df_clean[["age", "fare"]])
scaled_df = pd.DataFrame(scaled, columns=["age_z", "fare_z"])
print("\nAfter scaling:")
print(scaled_df.agg(["mean", "std"]))
print("\n(Exploratory only — the modeling pipeline performs its own train-only scaling.)")

print("\nEDA COMPLETE")
