"""
Zepto Analytics - Part B: Modeling
Continues from titanic_clean.csv (produced by 01_eda.py). Stratified split,
fit-on-train-only Pipeline/ColumnTransformer, 3 classifiers, imbalance
comparison (baseline vs class_weight vs SMOTE), GridSearchCV + OOB on RF,
regression side-task, saved joblib pipeline.
"""

from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, roc_curve, auc, mean_absolute_error,
    mean_squared_error, r2_score,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")


OUT_DIR = "analytics"
PLOT_DIR = os.path.join(OUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# ============ 1. LOAD (from committed CSV, not seaborn again) ============
df = pd.read_csv(os.path.join(OUT_DIR, "titanic_clean.csv"))
print("=" * 70)
print("1. LOAD CLEANED DATA")
print("=" * 70)
print("Shape:", df.shape)

X = df.drop(columns=["survived"])
y = df["survived"]

print("\nClass balance:")
print(y.value_counts(normalize=True))

# ============ 2. STRATIFIED SPLIT FIRST ============
print("\n" + "=" * 70)
print("2. STRATIFIED TRAIN/TEST SPLIT")
print("=" * 70)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print("Train class balance:", y_train.value_counts(normalize=True).to_dict())
print("Test class balance: ", y_test.value_counts(normalize=True).to_dict())

# ============ 3. PREPROCESSING PIPELINE (fit on train ONLY) ============
numeric_cols = ["age", "sibsp", "parch", "fare"]
categorical_cols = ["sex", "embarked"]

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]), numeric_cols),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]), categorical_cols),
])

# ============ 4. TRAIN 3 CLASSIFIERS ============
print("\n" + "=" * 70)
print("4. TRAIN 3 CLASSIFIERS")
print("=" * 70)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
}

results = {}
for name, model in models.items():
    pipe = Pipeline([("prep", preprocessor), ("clf", model)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)

    results[name] = {
        "pipeline": pipe,
        "cm": confusion_matrix(y_test, y_pred),
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "auc": auc(fpr, tpr),
        "fpr": fpr,
        "tpr": tpr,
    }

    print(f"\n--- {name} ---")
    print("Confusion matrix:")
    print(results[name]["cm"])
    print(f"  Accuracy : {results[name]['accuracy']:.4f}")
    print(f"  Precision: {results[name]['precision']:.4f}")
    print(f"  Recall   : {results[name]['recall']:.4f}")
    print(f"  F1       : {results[name]['f1']:.4f}")
    print(f"  AUC      : {results[name]['auc']:.4f}")

# Plot ROC curves side-by-side
plt.figure(figsize=(8, 6))
for name, r in results.items():
    plt.plot(r["fpr"], r["tpr"], label=f"{name} (AUC={r['auc']:.3f})")
plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC curves — all 3 classifiers")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "roc_curves.png"), dpi=100)
plt.close()

# ============ 5. DECISION TREE VISUALIZATION ============
print("\n" + "=" * 70)
print("5. DECISION TREE VISUALIZATION")
print("=" * 70)
dt_pipe = results["Decision Tree"]["pipeline"]
feature_names = dt_pipe.named_steps["prep"].get_feature_names_out()
plt.figure(figsize=(20, 10))
plot_tree(
    dt_pipe.named_steps["clf"],
    feature_names=feature_names,
    class_names=["Died", "Survived"],
    filled=True,
    fontsize=8,
)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "decision_tree.png"),
            dpi=100, bbox_inches="tight")
plt.close()
print(f"Saved decision tree plot to {PLOT_DIR}/decision_tree.png")

# ============ 6. COMPARISON TABLE ============
print("\n" + "=" * 70)
print("6. CLASSIFIER COMPARISON TABLE")
print("=" * 70)
comparison = pd.DataFrame({
    name: {
        "Accuracy": r["accuracy"],
        "Precision": r["precision"],
        "Recall": r["recall"],
        "F1": r["f1"],
        "AUC": r["auc"],
    } for name, r in results.items()
}).T
print(comparison.round(4))

# ============ 7. IMBALANCE COMPARISON ============
print("\n" + "=" * 70)
print("7. IMBALANCE COMPARISON (Logistic Regression)")
print("=" * 70)
print("Original class balance:", y.value_counts(normalize=True).to_dict())

# (a) baseline
pipe_a = Pipeline([("prep", preprocessor),
                   ("clf", LogisticRegression(max_iter=1000, random_state=42))])
pipe_a.fit(X_train, y_train)
pred_a = pipe_a.predict(X_test)

# (b) class_weight='balanced'
pipe_b = Pipeline([("prep", preprocessor),
                   ("clf", LogisticRegression(max_iter=1000, class_weight="balanced",
                                              random_state=42))])
pipe_b.fit(X_train, y_train)
pred_b = pipe_b.predict(X_test)

# (c) SMOTE on TRAINING FOLD ONLY
X_train_pre = preprocessor.fit_transform(X_train)
X_test_pre = preprocessor.transform(X_test)
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train_pre, y_train)
lr_smote = LogisticRegression(max_iter=1000, random_state=42)
lr_smote.fit(X_res, y_res)
pred_c = lr_smote.predict(X_test_pre)

for label, yp in [("baseline", pred_a),
                  ("class_weight='balanced'", pred_b),
                  ("SMOTE (train only)", pred_c)]:
    print(f"\n{label}:")
    print(f"  Precision: {precision_score(y_test, yp):.4f}")
    print(f"  Recall   : {recall_score(y_test, yp):.4f}")
    print(f"  F1       : {f1_score(y_test, yp):.4f}")

# ============ 8. GRIDSEARCHCV + OOB ON RANDOM FOREST ============
print("\n" + "=" * 70)
print("8. GRIDSEARCHCV (Random Forest, oob_score=True)")
print("=" * 70)
rf = RandomForestClassifier(oob_score=True, random_state=42, bootstrap=True)
rf_pipe = Pipeline([("prep", preprocessor), ("clf", rf)])
param_grid = {
    "clf__n_estimators": [50, 100],
    "clf__max_depth": [5, 10, None],
    "clf__max_features": ["sqrt", "log2"],
}
grid = GridSearchCV(rf_pipe, param_grid, cv=3, scoring="f1", n_jobs=-1)
grid.fit(X_train, y_train)
print("Best params:", grid.best_params_)
print(f"Best CV F1: {grid.best_score_:.4f}")
print(f"OOB score: {grid.best_estimator_.named_steps['clf'].oob_score_:.4f}")

# ============ 9. REGRESSION SIDE-TASK: PREDICT FARE ============
print("\n" + "=" * 70)
print("9. REGRESSION SIDE-TASK - PREDICT FARE")
print("=" * 70)
reg_df = df.dropna(subset=["fare"]).copy()
X_reg = reg_df.drop(columns=["fare"])
y_reg = reg_df["fare"]

Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

reg_pre = ColumnTransformer([
    ("num", Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
    ]), ["age", "sibsp", "parch", "pclass", "survived"]),
    ("cat", Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("oh", OneHotEncoder(handle_unknown="ignore")),
    ]), ["sex", "embarked"]),
])

reg_pipe = Pipeline([("prep", reg_pre), ("reg", LinearRegression())])
reg_pipe.fit(Xr_train, yr_train)
yr_pred = reg_pipe.predict(Xr_test)

mae = mean_absolute_error(yr_test, yr_pred)
rmse = np.sqrt(mean_squared_error(yr_test, yr_pred))
r2 = r2_score(yr_test, yr_pred)
n = len(yr_test)
p = Xr_test.shape[1]
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

print(f"MAE      : {mae:.4f}")
print(f"RMSE     : {rmse:.4f}")
print(f"R^2      : {r2:.4f}")
print(f"Adj R^2  : {adj_r2:.4f}")

# Residual plot
residuals = yr_test - yr_pred
plt.figure(figsize=(8, 5))
plt.scatter(yr_pred, residuals, alpha=0.5)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Predicted fare")
plt.ylabel("Residuals")
plt.title("Residual plot - regression on fare")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "residual_plot.png"), dpi=100)
plt.close()
print(f"Saved residual plot to {PLOT_DIR}/residual_plot.png")

# ============ 10. SAVE FULL PIPELINE ============
print("\n" + "=" * 70)
print("10. SAVE BEST PIPELINE")
print("=" * 70)
best_name = comparison["F1"].idxmax()
best_pipe = results[best_name]["pipeline"]
save_path = os.path.join(OUT_DIR, "best_pipeline.joblib")
joblib.dump(best_pipe, save_path)
print(f"Best classifier by F1: {best_name}")
print(f"Saved full pipeline to {save_path}")

# ============ 11. RELOAD TEST ============
print("\n" + "=" * 70)
print("11. RELOAD TEST - confirm pipeline works on raw data")
print("=" * 70)
loaded = joblib.load(save_path)
raw_sample = X_test.iloc[:5]
preds = loaded.predict(raw_sample)
print("Raw sample predictions:", preds.tolist())
print("(Pipeline correctly handles raw, unpreprocessed input.)")

print("\nMODELING COMPLETE")
