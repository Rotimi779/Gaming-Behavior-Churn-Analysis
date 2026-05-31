"""
Phase 3: Modeling & Comparison
World of Warcraft Player Churn Prediction Project

This script:
  1. Loads the stratified train/test split from Phase 2
  2. Trains four classifiers with class-imbalance handling:
       - Logistic Regression
       - Random Forest
       - XGBoost
       - LightGBM
  3. Compares them on ROC-AUC, Precision, Recall, F1
  4. Plots ROC curves and a metric comparison
  5. Selects the best model by ROC-AUC
  6. Runs detailed evaluation (classification report + confusion matrix)
  7. Tunes the best model with GridSearchCV (stratified 5-fold)
  8. Saves the tuned best model and all results

Run from the project root:
    python src/phase3_modeling.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (roc_auc_score, precision_score, recall_score,
                             f1_score, roc_curve, classification_report,
                             confusion_matrix)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths & config
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
SCREENSHOTS_DIR = os.path.join(PROJECT_ROOT, "screenshots")

for d in (MODELS_DIR, RESULTS_DIR, SCREENSHOTS_DIR):
    os.makedirs(d, exist_ok=True)

sns.set_style("whitegrid")
RANDOM_STATE = 42


def section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# ---------------------------------------------------------------------------
# Step 1: Load data
# ---------------------------------------------------------------------------
def load_data():
    section("STEP 1: LOAD TRAIN/TEST SPLIT")

    X_train = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(PROCESSED_DIR, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(PROCESSED_DIR, "y_train.csv")).iloc[:, 0]
    y_test = pd.read_csv(os.path.join(PROCESSED_DIR, "y_test.csv")).iloc[:, 0]

    print(f"X_train: {X_train.shape}   y_train churn rate: "
          f"{y_train.mean() * 100:.1f}%")
    print(f"X_test:  {X_test.shape}   y_test churn rate:  "
          f"{y_test.mean() * 100:.1f}%")
    print(f"Features: {X_train.shape[1]}")

    # Scaled copies for Logistic Regression (tree models don't need scaling)
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns
    )
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))

    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled


# ---------------------------------------------------------------------------
# Step 2: Train the four models
# ---------------------------------------------------------------------------
def train_models(X_train, X_test, y_train, y_test,
                  X_train_scaled, X_test_scaled):
    section("STEP 2: TRAIN FOUR MODELS")

    # Class imbalance: ~31.6% churn -> weight for XGBoost
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = n_neg / n_pos
    print(f"Class balance: {n_neg} retained / {n_pos} churned "
          f"-> scale_pos_weight = {scale_pos_weight:.2f}\n")

    models = {}
    predictions = {}

    # ---- Logistic Regression (uses scaled features) ----------------------
    lr = LogisticRegression(class_weight="balanced", max_iter=1000,
                            random_state=RANDOM_STATE)
    lr.fit(X_train_scaled, y_train)
    models["Logistic Regression"] = lr
    predictions["Logistic Regression"] = {
        "proba": lr.predict_proba(X_test_scaled)[:, 1],
        "pred": lr.predict(X_test_scaled),
    }
    print("  Trained: Logistic Regression")

    # ---- Random Forest ---------------------------------------------------
    rf = RandomForestClassifier(
        n_estimators=200, class_weight="balanced", max_depth=10,
        random_state=RANDOM_STATE, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    models["Random Forest"] = rf
    predictions["Random Forest"] = {
        "proba": rf.predict_proba(X_test)[:, 1],
        "pred": rf.predict(X_test),
    }
    print("  Trained: Random Forest")

    # ---- XGBoost ---------------------------------------------------------
    xgb = XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        scale_pos_weight=scale_pos_weight, random_state=RANDOM_STATE,
        eval_metric="logloss", n_jobs=-1
    )
    xgb.fit(X_train, y_train)
    models["XGBoost"] = xgb
    predictions["XGBoost"] = {
        "proba": xgb.predict_proba(X_test)[:, 1],
        "pred": xgb.predict(X_test),
    }
    print("  Trained: XGBoost")

    # ---- LightGBM --------------------------------------------------------
    lgbm = LGBMClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        class_weight="balanced", random_state=RANDOM_STATE,
        n_jobs=-1, verbose=-1
    )
    lgbm.fit(X_train, y_train)
    models["LightGBM"] = lgbm
    predictions["LightGBM"] = {
        "proba": lgbm.predict_proba(X_test)[:, 1],
        "pred": lgbm.predict(X_test),
    }
    print("  Trained: LightGBM")

    return models, predictions, scale_pos_weight


# ---------------------------------------------------------------------------
# Step 3: Compare models
# ---------------------------------------------------------------------------
def compare_models(predictions, y_test):
    section("STEP 3: MODEL COMPARISON")

    rows = []
    for name, pred in predictions.items():
        rows.append({
            "model": name,
            "roc_auc": roc_auc_score(y_test, pred["proba"]),
            "precision": precision_score(y_test, pred["pred"]),
            "recall": recall_score(y_test, pred["pred"]),
            "f1": f1_score(y_test, pred["pred"]),
        })
    comparison = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)

    print("All metrics (sorted by ROC-AUC):\n")
    print(comparison.round(4).to_string(index=False))

    comparison.to_csv(os.path.join(RESULTS_DIR,
                      "model_comparison_results.csv"), index=False)
    return comparison


# ---------------------------------------------------------------------------
# Step 4: ROC curves + metric comparison charts
# ---------------------------------------------------------------------------
def plot_comparisons(predictions, y_test, comparison):
    section("STEP 4: COMPARISON CHARTS")

    # ---- ROC curves ------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 8))
    for name, pred in predictions.items():
        fpr, tpr, _ = roc_curve(y_test, pred["proba"])
        auc = roc_auc_score(y_test, pred["proba"])
        ax.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random (AUC = 0.500)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves - All Models")
    ax.legend(loc="lower right")
    plt.tight_layout()
    out = os.path.join(SCREENSHOTS_DIR, "12_roc_curves.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved chart: {out}")

    # ---- Metric comparison bar chart -------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6))
    metrics = ["roc_auc", "precision", "recall", "f1"]
    x = np.arange(len(comparison))
    width = 0.2
    colors = ["#3498db", "#2ecc71", "#e67e22", "#9b59b6"]
    for i, metric in enumerate(metrics):
        ax.bar(x + i * width, comparison[metric], width,
               label=metric.upper(), color=colors[i])
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(comparison["model"], rotation=15)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison Across Metrics")
    ax.legend()
    ax.set_ylim(0, 1)
    plt.tight_layout()
    out = os.path.join(SCREENSHOTS_DIR, "13_model_comparison.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved chart: {out}")


# ---------------------------------------------------------------------------
# Step 5: Detailed evaluation of the best model
# ---------------------------------------------------------------------------
def evaluate_best(comparison, predictions, y_test):
    section("STEP 5: DETAILED EVALUATION OF BEST MODEL")

    best_name = comparison.iloc[0]["model"]
    print(f"Best model by ROC-AUC: {best_name}\n")

    pred = predictions[best_name]
    print("Classification report:")
    print(classification_report(y_test, pred["pred"],
                                target_names=["Retained", "Churned"]))

    cm = confusion_matrix(y_test, pred["pred"])
    tn, fp, fn, tp = cm.ravel()
    print("Confusion matrix breakdown:")
    print(f"  True Negatives  (retained, predicted retained): {tn}")
    print(f"  False Positives (retained, predicted churned):  {fp}")
    print(f"  False Negatives (churned, predicted retained):  {fn}")
    print(f"  True Positives  (churned, predicted churned):   {tp}")
    print(f"\n  Of {tp + fn} actual churners, the model caught {tp} "
          f"({tp / (tp + fn) * 100:.1f}%).")
    print(f"  Of {tp + fp} flagged as churn, {tp} actually churned "
          f"({tp / (tp + fp) * 100:.1f}%).")

    # ---- Confusion matrix chart ------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Retained", "Churned"],
                yticklabels=["Retained", "Churned"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix - {best_name}")
    plt.tight_layout()
    out = os.path.join(SCREENSHOTS_DIR, "14_confusion_matrix.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved chart: {out}")

    return best_name


# ---------------------------------------------------------------------------
# Step 6: Hyperparameter tuning of the best model
# ---------------------------------------------------------------------------
def tune_best_model(best_name, models, X_train, X_test, y_train, y_test,
                    X_train_scaled, X_test_scaled, scale_pos_weight):
    section("STEP 6: HYPERPARAMETER TUNING")
    print(f"Tuning {best_name} with GridSearchCV (stratified 5-fold)\n")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    if best_name == "Logistic Regression":
        estimator = LogisticRegression(class_weight="balanced",
                                       max_iter=1000,
                                       random_state=RANDOM_STATE)
        grid = {"C": [0.01, 0.1, 1, 10], "penalty": ["l2"]}
        X_tr, X_te = X_train_scaled, X_test_scaled

    elif best_name == "Random Forest":
        estimator = RandomForestClassifier(class_weight="balanced",
                                           random_state=RANDOM_STATE,
                                           n_jobs=-1)
        grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [5, 10, 15],
            "min_samples_split": [2, 5],
        }
        X_tr, X_te = X_train, X_test

    elif best_name == "XGBoost":
        estimator = XGBClassifier(scale_pos_weight=scale_pos_weight,
                                  random_state=RANDOM_STATE,
                                  eval_metric="logloss", n_jobs=-1)
        grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.05, 0.1, 0.2],
        }
        X_tr, X_te = X_train, X_test

    else:  # LightGBM
        estimator = LGBMClassifier(class_weight="balanced",
                                   random_state=RANDOM_STATE,
                                   n_jobs=-1, verbose=-1)
        grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.05, 0.1, 0.2],
        }
        X_tr, X_te = X_train, X_test

    search = GridSearchCV(estimator, grid, cv=cv, scoring="roc_auc",
                          n_jobs=-1, verbose=0)
    search.fit(X_tr, y_train)

    print(f"Best CV ROC-AUC:  {search.best_score_:.4f}")
    print(f"Best parameters:  {search.best_params_}")

    tuned = search.best_estimator_
    tuned_proba = tuned.predict_proba(X_te)[:, 1]
    tuned_test_auc = roc_auc_score(y_test, tuned_proba)
    print(f"Tuned model test ROC-AUC: {tuned_test_auc:.4f}")

    # Save the tuned best model
    joblib.dump(tuned, os.path.join(MODELS_DIR, "best_model.pkl"))
    with open(os.path.join(MODELS_DIR, "best_model_info.txt"), "w") as f:
        f.write(f"Best model: {best_name}\n")
        f.write(f"Best CV ROC-AUC: {search.best_score_:.4f}\n")
        f.write(f"Tuned test ROC-AUC: {tuned_test_auc:.4f}\n")
        f.write(f"Best parameters: {search.best_params_}\n")
        f.write(f"Uses scaled features: "
                f"{best_name == 'Logistic Regression'}\n")
    print(f"\n  Saved: models/best_model.pkl")
    print(f"  Saved: models/best_model_info.txt")

    return tuned, tuned_test_auc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    section("PHASE 3: MODELING & COMPARISON")
    print("World of Warcraft Player Churn Prediction Project")

    (X_train, X_test, y_train, y_test,
     X_train_scaled, X_test_scaled) = load_data()

    models, predictions, scale_pos_weight = train_models(
        X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled
    )

    comparison = compare_models(predictions, y_test)
    plot_comparisons(predictions, y_test, comparison)
    best_name = evaluate_best(comparison, predictions, y_test)
    tuned, tuned_test_auc = tune_best_model(
        best_name, models, X_train, X_test, y_train, y_test,
        X_train_scaled, X_test_scaled, scale_pos_weight
    )

    section("PHASE 3 COMPLETE")
    print(f"Best model: {best_name}")
    print(f"Tuned test ROC-AUC: {tuned_test_auc:.4f}")
    print("\nOutputs:")
    print("  models/best_model.pkl")
    print("  models/scaler.pkl")
    print("  models/best_model_info.txt")
    print("  results/model_comparison_results.csv")
    print("  screenshots/12_roc_curves.png")
    print("  screenshots/13_model_comparison.png")
    print("  screenshots/14_confusion_matrix.png")


if __name__ == "__main__":
    main()
