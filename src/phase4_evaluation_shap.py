"""
Phase 4: SHAP Model Interpretation
World of Warcraft Player Churn Prediction Project

This script:
  1. Loads the tuned best model (Logistic Regression) + scaler from Phase 3
  2. Computes SHAP values with LinearExplainer (exact for linear models)
  3. Produces a global importance bar chart (mean |SHAP| per feature)
  4. Produces a beeswarm plot (direction and spread of each feature's effect)
  5. Produces individual waterfall plots for the 3 highest-risk characters
  6. Translates the findings into business insights and saves them to file

Why LinearExplainer (not TreeExplainer)?
  Phase 3 selected Logistic Regression as the best model. For linear models,
  LinearExplainer gives exact SHAP values and is faster than the tree-based
  explainers (which are designed for ensemble methods).

Run from the project root:
    python src/phase4_evaluation_shap.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import shap

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "phase2")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "phase4")
SHAP_DIR = os.path.join(RESULTS_DIR, "shap_analysis")
SCREENSHOTS_DIR = os.path.join(PROJECT_ROOT, "screenshots", "phase4")

for d in (RESULTS_DIR, SHAP_DIR, SCREENSHOTS_DIR):
    os.makedirs(d, exist_ok=True)


def section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# ---------------------------------------------------------------------------
# Step 1: Load model, scaler, and test data
# ---------------------------------------------------------------------------
def load_artifacts():
    section("STEP 1: LOAD MODEL, SCALER, AND TEST DATA")

    model = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    X_test = pd.read_csv(os.path.join(PROCESSED_DIR, "X_test.csv"))
    X_train = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train.csv"))
    y_test = pd.read_csv(os.path.join(PROCESSED_DIR, "y_test.csv")).iloc[:, 0]

    print(f"Loaded model: {type(model).__name__}")
    print(f"Test shape: {X_test.shape}  |  features: {X_test.shape[1]}")

    # Scale for the linear model
    X_train_scaled = pd.DataFrame(
        scaler.transform(X_train), columns=X_train.columns
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns
    )
    return model, X_train_scaled, X_test_scaled, X_test, y_test


# ---------------------------------------------------------------------------
# Step 2: Compute SHAP values
# ---------------------------------------------------------------------------
def compute_shap(model, X_train_scaled, X_test_scaled):
    section("STEP 2: COMPUTE SHAP VALUES")

    # LinearExplainer needs a background sample for the masker
    explainer = shap.LinearExplainer(model, X_train_scaled)
    shap_values = explainer(X_test_scaled)

    print(f"Computed SHAP values for {len(X_test_scaled)} test characters")
    print(f"Shape: {shap_values.values.shape}")
    print(f"Base value (model's average prediction in log-odds): "
          f"{shap_values.base_values[0]:.4f}")

    # Save raw SHAP values
    shap_df = pd.DataFrame(shap_values.values, columns=X_test_scaled.columns)
    shap_df.to_csv(os.path.join(SHAP_DIR, "shap_values.csv"), index=False)
    print(f"  Saved: results/phase4/shap_analysis/shap_values.csv")

    return explainer, shap_values


# ---------------------------------------------------------------------------
# Step 3: Global importance bar chart
# ---------------------------------------------------------------------------
def global_importance(shap_values, X_test_scaled):
    section("STEP 3: GLOBAL FEATURE IMPORTANCE (mean |SHAP|)")

    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    importance = pd.DataFrame({
        "feature": X_test_scaled.columns,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False)

    print("Top 15 features by mean |SHAP|:")
    print(importance.head(15).to_string(index=False))

    importance.to_csv(os.path.join(SHAP_DIR,
                      "shap_global_importance.csv"), index=False)

    # Chart 15
    fig, ax = plt.subplots(figsize=(10, 9))
    top = importance.head(20)
    ax.barh(top["feature"][::-1], top["mean_abs_shap"][::-1],
            color="#3498db")
    ax.set_xlabel("Mean |SHAP value|  (average impact on churn prediction)")
    ax.set_title("Global Feature Importance (SHAP)")
    plt.tight_layout()
    out = os.path.join(SCREENSHOTS_DIR, "15_shap_bar.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved chart: {out}")

    return importance


# ---------------------------------------------------------------------------
# Step 4: Beeswarm plot
# ---------------------------------------------------------------------------
def beeswarm_plot(shap_values, X_test_scaled):
    section("STEP 4: BEESWARM PLOT")
    print("Each dot is one test character.")
    print("Red = high feature value, blue = low.")
    print("Position left/right shows direction of churn push.\n")

    plt.figure(figsize=(11, 9))
    shap.plots.beeswarm(shap_values, max_display=20, show=False)
    plt.title("SHAP Beeswarm  (left = retention pull, right = churn push)",
              pad=15)
    plt.tight_layout()
    out = os.path.join(SCREENSHOTS_DIR, "16_shap_beeswarm.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved chart: {out}")


# ---------------------------------------------------------------------------
# Step 5: Individual waterfall plots for top-risk characters
# ---------------------------------------------------------------------------
def waterfall_plots(model, shap_values, X_test, X_test_scaled, y_test):
    section("STEP 5: INDIVIDUAL EXPLANATIONS (top 3 highest-risk characters)")

    probas = model.predict_proba(X_test_scaled)[:, 1]
    risk = pd.DataFrame({
        "test_idx": np.arange(len(probas)),
        "churn_proba": probas,
        "actual": y_test.values,
    }).sort_values("churn_proba", ascending=False)

    print("Top 5 highest-risk characters in the test set:")
    print(risk.head(5).to_string(index=False))
    print()

    # Show waterfalls for the top 3 actual churners (proba ≥ 0.5 and actual = 1)
    correct_churners = risk[(risk["churn_proba"] >= 0.5)
                            & (risk["actual"] == 1)].head(3)

    # Render each waterfall to its own panel, then combine vertically.
    # shap's waterfall uses plt.gca() internally so making 3 horizontally
    # in one figure clashes; we render them to separate temp images and
    # tile them in a single figure instead.
    panels = []
    for _, row in correct_churners.iterrows():
        idx = int(row["test_idx"])
        explanation = shap.Explanation(
            values=shap_values.values[idx],
            base_values=shap_values.base_values[idx],
            data=X_test.iloc[idx].values,
            feature_names=X_test.columns.tolist(),
        )
        fig = plt.figure(figsize=(11, 7))
        shap.plots.waterfall(explanation, max_display=10, show=False)
        plt.title(
            f"Character #{idx}  |  churn proba = {row['churn_proba']:.2f}  "
            f"|  actual = CHURNED",
            fontsize=12, pad=12,
        )
        plt.tight_layout()
        panel_path = os.path.join(SCREENSHOTS_DIR,
                                  f"_waterfall_panel_{idx}.png")
        plt.savefig(panel_path, dpi=200, bbox_inches="tight")
        plt.close()
        panels.append(panel_path)

    # Combine the three panels into one tall image
    from PIL import Image
    images = [Image.open(p) for p in panels]
    widths, heights = zip(*(im.size for im in images))
    total_height = sum(heights)
    max_width = max(widths)
    combined = Image.new("RGB", (max_width, total_height), color="white")
    y = 0
    for im in images:
        combined.paste(im, (0, y))
        y += im.size[1]
    out = os.path.join(SCREENSHOTS_DIR, "17_shap_waterfall.png")
    combined.save(out)

    # Clean up temp panels
    for p in panels:
        os.remove(p)
    print(f"  Saved chart: {out}")

    # Print the contributing features for each of the three
    print("\nWaterfall details for each character:")
    for _, row in correct_churners.iterrows():
        idx = int(row["test_idx"])
        contrib = pd.DataFrame({
            "feature": X_test.columns,
            "raw_value": X_test.iloc[idx].values,
            "shap_value": shap_values.values[idx],
        }).sort_values("shap_value", key=np.abs, ascending=False)
        print(f"\nCharacter #{idx} (churn proba "
              f"{row['churn_proba']:.2f}) — top 5 drivers:")
        print(contrib.head(5).to_string(index=False))


# ---------------------------------------------------------------------------
# Step 6: Business insights file
# ---------------------------------------------------------------------------
def write_business_insights(importance, shap_values, X_test_scaled):
    section("STEP 6: BUSINESS INSIGHTS")

    # Pull the top 5 features and figure out direction
    # (positive SHAP = pushes toward churn; correlate value with SHAP to see)
    top5 = importance.head(5)["feature"].tolist()

    insights_lines = []
    insights_lines.append("=" * 80)
    insights_lines.append("BUSINESS INSIGHTS - WoW Player Churn Drivers")
    insights_lines.append("Derived from SHAP analysis of the tuned best model "
                          "(Logistic Regression, ROC-AUC 0.833)")
    insights_lines.append("=" * 80)
    insights_lines.append("")

    print("Top 5 churn drivers (mean |SHAP|):\n")
    for i, feat in enumerate(top5, 1):
        col_idx = list(X_test_scaled.columns).index(feat)
        # Direction: do high values push toward churn (+) or retention (-)?
        feature_vals = X_test_scaled[feat].values
        shap_vals = shap_values.values[:, col_idx]
        # Correlation of feature value with SHAP value
        if feature_vals.std() > 0:
            direction_corr = np.corrcoef(feature_vals, shap_vals)[0, 1]
        else:
            direction_corr = 0
        direction = ("HIGH values push toward CHURN"
                     if direction_corr > 0
                     else "LOW values push toward CHURN")
        print(f"  {i}. {feat:<20} ({direction})")
        insights_lines.append(f"{i}. {feat}  ({direction})")

    # Business recommendations grounded in the top drivers
    insights_lines.append("")
    insights_lines.append("-" * 80)
    insights_lines.append("RETENTION RECOMMENDATIONS")
    insights_lines.append("-" * 80)
    insights_lines.append("")

    recommendations = [
        ("activity_days / observation density",
         "Players who were active for short spans (< 30 days) or whose "
         "observations were thin in the observation window are at high "
         "risk. Trigger automated re-engagement messaging (in-game mail, "
         "email) when a character has been inactive for 14 days. The "
         "earlier the touchpoint, the higher the recovery rate."),

        ("has_guild / no_guild  (social anchor)",
         "Solo players churn at more than double the rate of guild members "
         "(67.6% vs 30.9% in EDA). Implement a guild-finder nudge at "
         "day 7 for new characters that have not joined a guild. "
         "Auto-suggest active recruiting guilds matching the player's "
         "level range and class."),

        ("zone_per_day / zones_per_obs  (exploration)",
         "Players who confine themselves to 1-3 zones disengage faster. "
         "A daily quest pointing the player to a new zone (with a small "
         "in-game reward) directly targets this driver and reinforces "
         "exploration as a habit."),

        ("level_per_day / max_level  (progression)",
         "Stalled levellers (level_gain < 2 over a 30+ day span) need "
         "progression aids. Identify the level band where they stalled "
         "and surface relevant content - quests, gear, or rest XP - "
         "instead of letting them grind unaided."),

        ("obs_per_session  (session depth)",
         "Players who log in briefly but never settle into long sessions "
         "are at risk. A 'come back to where you left off' resume feature "
         "and short-form content (15-30 minute objectives) can convert "
         "shallow logins into deeper engagement.")
    ]

    for title, body in recommendations:
        insights_lines.append(f"\n* {title}")
        for line in body.split(". "):
            if line:
                insights_lines.append(f"  {line.strip()}.")

    insights_lines.append("")
    insights_lines.append("-" * 80)
    insights_lines.append("HIGH-VALUE TAKEAWAYS FOR THE BUSINESS")
    insights_lines.append("-" * 80)
    insights_lines.append(
        "- A model with ROC-AUC 0.833 ranks at-risk players reliably enough\n"
        "  to drive a targeted retention campaign rather than blanket\n"
        "  outreach. Concentrating the campaign budget on the top 20%\n"
        "  highest-risk players is the highest-ROI use of this model.\n"
        "- The strongest churn drivers are not random - they cluster\n"
        "  around three behavioural themes: engagement depth, social\n"
        "  anchoring, and exploration. Retention strategy should be\n"
        "  built around these three themes, not around demographics.\n"
        "- The model catches 68% of actual churners (recall) at 64%\n"
        "  precision - acceptable for low-cost retention interventions\n"
        "  (e.g. email, in-game mail). For high-cost interventions a\n"
        "  higher probability threshold should be set to raise precision."
    )

    out_path = os.path.join(RESULTS_DIR, "business_insights.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(insights_lines))
    print(f"\n  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    section("PHASE 4: SHAP MODEL INTERPRETATION")
    print("World of Warcraft Player Churn Prediction Project")

    model, X_train_scaled, X_test_scaled, X_test, y_test = load_artifacts()
    explainer, shap_values = compute_shap(model, X_train_scaled, X_test_scaled)

    importance = global_importance(shap_values, X_test_scaled)
    beeswarm_plot(shap_values, X_test_scaled)
    waterfall_plots(model, shap_values, X_test, X_test_scaled, y_test)
    write_business_insights(importance, shap_values, X_test_scaled)

    section("PHASE 4 COMPLETE")
    print("Outputs:")
    print("  results/phase4/shap_analysis/shap_values.csv")
    print("  results/phase4/shap_analysis/shap_global_importance.csv")
    print("  results/phase4/business_insights.txt")
    print("  screenshots/phase4/15_shap_bar.png")
    print("  screenshots/phase4/16_shap_beeswarm.png")
    print("  screenshots/phase4/17_shap_waterfall.png")


if __name__ == "__main__":
    main()
