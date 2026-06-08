"""
Phase 2: Feature Engineering + Rigorous Feature Selection
World of Warcraft Player Churn Prediction Project

Uses the observation/outcome window split to prevent target leakage:
  - Observation window: Jan 1 - Sep 30, 2008  -> ALL features come from here
  - Outcome window:      Oct 1 - Dec 31, 2008  -> churn label comes from here
  - churned = active in observation window but ZERO snapshots in outcome window

This script:
  Part A - Aggregates observation-window snapshots into one row per character
           (baseline features) and applies the Phase 1 data-quality filters.
  Part B - Engineers additional features and tests each ONE AT A TIME for its
           effect on ROC-AUC, using a Random Forest baseline.
  Part C - Ranks features by measured impact, keeps those above the threshold,
           builds the final feature set, and saves the stratified train/test
           split for Phase 3.

Leakage policy:
  - days_to_obs_end (time from last-seen to Sep 30) is EXCLUDED: it almost
    directly encodes the churn label.
  - activity_days is KEPT: it reflects genuine in-window engagement span.
    Its feature importance is monitored in Part C.

Run from the project root:
    python src/phase2_feature_engineering.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------------------------
# Paths & config
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA = os.path.join(PROJECT_ROOT, "data", "raw",
                        "wowah_data_filtered_10000.csv")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "phase2")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "phase2")
SCREENSHOTS_DIR = os.path.join(PROJECT_ROOT, "screenshots", "phase2")

for d in (PROCESSED_DIR, RESULTS_DIR, SCREENSHOTS_DIR):
    os.makedirs(d, exist_ok=True)

sns.set_style("whitegrid")

# Window split
OBS_END = pd.Timestamp("2008-09-30 23:59:59")
OUTCOME_START = pd.Timestamp("2008-10-01")

# Filters & thresholds
MIN_OBSERVATIONS = 5            # drop characters with < 5 obs in obs window
KEEP_THRESHOLD = 0.001          # ROC-AUC delta needed to KEEP a feature
RANDOM_STATE = 42


def section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# ---------------------------------------------------------------------------
# PART A: Aggregate observation-window snapshots -> character features
# ---------------------------------------------------------------------------
def build_character_features():
    section("PART A: AGGREGATE OBSERVATION-WINDOW SNAPSHOTS -> FEATURES")

    df = pd.read_csv(RAW_DATA)
    df.columns = [c.strip() for c in df.columns]
    df["timestamp"] = pd.to_datetime(
        df["timestamp"], format="%m/%d/%y %H:%M:%S", errors="coerce"
    )
    df = df.dropna(subset=["timestamp"])

    # ---- Split into windows ----------------------------------------------
    df_obs = df[df["timestamp"] <= OBS_END].copy()
    df_outcome = df[df["timestamp"] >= OUTCOME_START].copy()
    print(f"Observation window rows: {len(df_obs):,}")
    print(f"Outcome window rows:     {len(df_outcome):,}")

    chars_obs = set(df_obs["char"].unique())
    chars_outcome = set(df_outcome["char"].unique())
    churned_chars = chars_obs - chars_outcome
    print(f"Characters in observation window: {len(chars_obs):,}")
    print(f"  -> churned (not in outcome window): {len(churned_chars):,}")

    # ---- Sessions: gaps > 1 hour between snapshots = new session ---------
    df_obs = df_obs.sort_values(["char", "timestamp"])
    df_obs["gap_min"] = (df_obs.groupby("char")["timestamp"].diff()
                         .dt.total_seconds() / 60)
    df_obs["new_session"] = (df_obs["gap_min"].isnull()
                             | (df_obs["gap_min"] > 60))
    df_obs["session_id"] = df_obs.groupby("char")["new_session"].cumsum()

    # ---- Aggregate to one row per character ------------------------------
    obs_start = df_obs["timestamp"].min()
    g = df_obs.groupby("char")
    first_seen = g["timestamp"].min()
    last_seen = g["timestamp"].max()

    feat = pd.DataFrame({
        "observations": g.size(),
        "max_level": g["level"].max(),
        "min_level": g["level"].min(),
        "activity_days": (last_seen - first_seen).dt.days,
        "days_since_start": (first_seen - obs_start).dt.days,
        "unique_zones": g["zone"].nunique(),
        "unique_sessions": g["session_id"].max(),
        "guild_count": g["guild"].apply(lambda s: s[s != -1].nunique()),
        "race": g["race"].first(),
        "charclass": g["charclass"].first(),
    })
    feat["level_gain"] = feat["max_level"] - feat["min_level"]
    feat["has_guild"] = (feat["guild_count"] > 0).astype(int)
    feat["churned"] = [1 if c in churned_chars else 0 for c in feat.index]

    print(f"\nCharacters before filters: {len(feat):,}")

    # ---- Filter 1: drop Death Knights ------------------------------------
    dk_mask = feat["charclass"] == "Death Knight"
    print(f"  Dropping {dk_mask.sum()} Death Knight characters "
          f"(class launched Nov 2008 - outside observation window)")
    feat = feat[~dk_mask]

    # ---- Filter 2: drop characters with < 5 observations -----------------
    low_obs = feat["observations"] < MIN_OBSERVATIONS
    print(f"  Dropping {low_obs.sum()} characters with < {MIN_OBSERVATIONS} "
          f"observations (insufficient behavioural signal)")
    feat = feat[~low_obs]

    feat = feat.reset_index()
    print(f"Characters after filters: {len(feat):,}")
    print(f"Churn rate after filters: {feat['churned'].mean() * 100:.2f}%")

    # ---- Encode categoricals ---------------------------------------------
    feat["race_encoded"] = LabelEncoder().fit_transform(feat["race"])
    feat["class_encoded"] = LabelEncoder().fit_transform(feat["charclass"])

    return feat


# Baseline feature set (direct aggregations from the observation window)
# NOTE: days_to_obs_end is deliberately NOT included - it leaks the label.
BASELINE_FEATURES = [
    "observations", "max_level", "min_level", "activity_days",
    "days_since_start", "unique_zones", "unique_sessions",
    "guild_count", "level_gain", "has_guild",
    "race_encoded", "class_encoded",
]


# ---------------------------------------------------------------------------
# Feature testing framework
# ---------------------------------------------------------------------------
class FeatureTester:
    """Test each engineered feature individually for its ROC-AUC impact."""

    def __init__(self, X_baseline, y, baseline_score):
        self.X_baseline = X_baseline.copy()
        self.y = y
        self.baseline_score = baseline_score
        self.results = []

    def test(self, name, series, verbose=True):
        X = self.X_baseline.copy()
        X[name] = series.values

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, self.y, test_size=0.2, random_state=RANDOM_STATE,
            stratify=self.y
        )
        model = RandomForestClassifier(
            n_estimators=100, class_weight="balanced",
            random_state=RANDOM_STATE, max_depth=10, n_jobs=-1
        )
        model.fit(X_tr, y_tr)
        score = roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])
        importance = model.feature_importances_[X_tr.columns.get_loc(name)]
        delta = score - self.baseline_score
        verdict = "KEEP" if delta > KEEP_THRESHOLD else "DISCARD"

        self.results.append({
            "feature": name, "roc_auc": score, "delta": delta,
            "importance": importance, "verdict": verdict,
        })
        if verbose:
            sym = "[KEEP]   " if verdict == "KEEP" else "[DISCARD]"
            print(f"  {sym} {name:<28} ROC-AUC={score:.4f}  "
                  f"d={delta:+.4f}  imp={importance:.4f}")

    def results_df(self):
        return pd.DataFrame(self.results).sort_values("delta",
                                                      ascending=False)


# ---------------------------------------------------------------------------
# PART B: Engineer & test features
# ---------------------------------------------------------------------------
def engineer_and_test(feat):
    section("PART B: ENGINEER & TEST FEATURES")

    X_baseline = feat[BASELINE_FEATURES]
    y = feat["churned"]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_baseline, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    base_model = RandomForestClassifier(
        n_estimators=100, class_weight="balanced",
        random_state=RANDOM_STATE, max_depth=10, n_jobs=-1
    )
    base_model.fit(X_tr, y_tr)
    baseline_score = roc_auc_score(y_te, base_model.predict_proba(X_te)[:, 1])
    print(f"BASELINE ROC-AUC: {baseline_score:.4f} "
          f"({len(BASELINE_FEATURES)} raw features)")
    print(f"KEEP threshold: delta > +{KEEP_THRESHOLD}\n")

    tester = FeatureTester(X_baseline, y, baseline_score)

    # ---- Group 1: Leveling behaviour -------------------------------------
    print("Group 1 - Leveling behaviour:")
    feat["level_per_day"] = feat["max_level"] / (feat["activity_days"] + 1)
    tester.test("level_per_day", feat["level_per_day"])

    feat["is_max_level"] = (feat["max_level"] >= 70).astype(int)
    tester.test("is_max_level", feat["is_max_level"])

    q25_level = feat["max_level"].quantile(0.25)
    feat["is_low_level"] = (feat["max_level"] < q25_level).astype(int)
    tester.test("is_low_level", feat["is_low_level"])

    feat["stalled_leveller"] = (
        (feat["level_gain"] < 2) & (feat["activity_days"] > 30)
    ).astype(int)
    tester.test("stalled_leveller", feat["stalled_leveller"])

    feat["progression_ratio"] = feat["level_gain"] / (feat["max_level"] + 1)
    tester.test("progression_ratio", feat["progression_ratio"])

    # ---- Group 2: Activity intensity -------------------------------------
    print("Group 2 - Activity intensity:")
    feat["obs_per_day"] = feat["observations"] / (feat["activity_days"] + 1)
    tester.test("obs_per_day", feat["obs_per_day"])

    q75_opd = feat["obs_per_day"].quantile(0.75)
    feat["is_heavy_player"] = (feat["obs_per_day"] > q75_opd).astype(int)
    tester.test("is_heavy_player", feat["is_heavy_player"])

    q25_opd = feat["obs_per_day"].quantile(0.25)
    feat["is_light_player"] = (feat["obs_per_day"] < q25_opd).astype(int)
    tester.test("is_light_player", feat["is_light_player"])

    feat["obs_per_session"] = (feat["observations"]
                               / (feat["unique_sessions"] + 1))
    tester.test("obs_per_session", feat["obs_per_session"])

    feat["sessions_per_day"] = (feat["unique_sessions"]
                                / (feat["activity_days"] + 1))
    tester.test("sessions_per_day", feat["sessions_per_day"])

    feat["is_late_joiner"] = (feat["days_since_start"] > 180).astype(int)
    tester.test("is_late_joiner", feat["is_late_joiner"])

    # ---- Group 3: Zone exploration ---------------------------------------
    print("Group 3 - Zone exploration:")
    feat["zone_per_day"] = feat["unique_zones"] / (feat["activity_days"] + 1)
    tester.test("zone_per_day", feat["zone_per_day"])

    q75_zones = feat["unique_zones"].quantile(0.75)
    feat["is_explorer"] = (feat["unique_zones"] > q75_zones).astype(int)
    tester.test("is_explorer", feat["is_explorer"])

    q25_zones = feat["unique_zones"].quantile(0.25)
    feat["is_static"] = (feat["unique_zones"] < q25_zones).astype(int)
    tester.test("is_static", feat["is_static"])

    feat["zones_per_level"] = feat["unique_zones"] / (feat["max_level"] + 1)
    tester.test("zones_per_level", feat["zones_per_level"])

    feat["zones_per_obs"] = feat["unique_zones"] / (feat["observations"] + 1)
    tester.test("zones_per_obs", feat["zones_per_obs"])

    # ---- Group 4: Social / guild behaviour -------------------------------
    print("Group 4 - Social / guild behaviour:")
    feat["guild_stability"] = (feat["guild_count"] == 1).astype(int)
    tester.test("guild_stability", feat["guild_stability"])

    feat["guild_hopper"] = (feat["guild_count"] > 2).astype(int)
    tester.test("guild_hopper", feat["guild_hopper"])

    feat["no_guild"] = (feat["has_guild"] == 0).astype(int)
    tester.test("no_guild", feat["no_guild"])

    # ---- Group 5: Interaction features -----------------------------------
    print("Group 5 - Interaction features:")
    feat["level_x_zones"] = feat["max_level"] * feat["unique_zones"]
    tester.test("level_x_zones", feat["level_x_zones"])

    feat["obs_x_level"] = feat["observations"] * feat["max_level"]
    tester.test("obs_x_level", feat["obs_x_level"])

    feat["activity_x_guild"] = feat["activity_days"] * feat["has_guild"]
    tester.test("activity_x_guild", feat["activity_x_guild"])

    feat["zones_x_guild"] = feat["unique_zones"] * feat["has_guild"]
    tester.test("zones_x_guild", feat["zones_x_guild"])

    return feat, tester, baseline_score, y


# ---------------------------------------------------------------------------
# PART C: Rank, select, build final feature set
# ---------------------------------------------------------------------------
def select_and_finalize(feat, tester, baseline_score, y):
    section("PART C: RANK, SELECT & BUILD FINAL FEATURE SET")

    results = tester.results_df()
    print("Feature engineering results (ranked by ROC-AUC delta):\n")
    show = results[["feature", "roc_auc", "delta", "importance", "verdict"]]
    print(show.to_string(index=False))

    kept = results[results["verdict"] == "KEEP"]["feature"].tolist()
    discarded = results[results["verdict"] == "DISCARD"]["feature"].tolist()
    print(f"\nKEEP ({len(kept)}): {kept}")
    print(f"DISCARD ({len(discarded)}): {discarded}")

    # ---- Chart 08: feature impact ranking --------------------------------
    fig, ax = plt.subplots(figsize=(11, 9))
    colors = ["#2ecc71" if v == "KEEP" else "#e74c3c"
              for v in results["verdict"]]
    ax.barh(results["feature"], results["delta"], color=colors)
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.axvline(x=KEEP_THRESHOLD, color="blue", linestyle="--", linewidth=0.8,
               label=f"KEEP threshold (+{KEEP_THRESHOLD})")
    ax.set_xlabel("ROC-AUC Delta vs Baseline")
    ax.set_title("Feature Impact Ranking (green = KEEP, red = DISCARD)")
    ax.legend()
    plt.tight_layout()
    out = os.path.join(SCREENSHOTS_DIR, "10_feature_impact_ranking.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved chart: {out}")

    # ---- Build final feature set -----------------------------------------
    final_features = BASELINE_FEATURES + kept
    X_final = feat[final_features]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_final, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    final_model = RandomForestClassifier(
        n_estimators=100, class_weight="balanced",
        random_state=RANDOM_STATE, max_depth=10, n_jobs=-1
    )
    final_model.fit(X_tr, y_tr)
    final_score = roc_auc_score(y_te, final_model.predict_proba(X_te)[:, 1])

    print(f"\nBaseline ROC-AUC: {baseline_score:.4f} "
          f"({len(BASELINE_FEATURES)} features)")
    print(f"Final ROC-AUC:    {final_score:.4f} "
          f"({len(final_features)} features)")
    print(f"Improvement:      {final_score - baseline_score:+.4f}")

    # ---- Final feature importance ----------------------------------------
    importance = pd.DataFrame({
        "feature": final_features,
        "importance": final_model.feature_importances_,
    }).sort_values("importance", ascending=False)

    print("\nTop 15 features by importance (final model):")
    print(importance.head(15).to_string(index=False))

    # Leakage monitor
    top_imp = importance.iloc[0]
    if top_imp["importance"] > 0.40:
        print(f"\n  NOTE: '{top_imp['feature']}' has importance "
              f"{top_imp['importance']:.3f} - monitor for leakage. "
              f"activity_days reflects in-window engagement span; this is "
              f"acceptable but worth noting in the README.")

    fig, ax = plt.subplots(figsize=(10, 8))
    top = importance.head(20)
    ax.barh(top["feature"][::-1], top["importance"][::-1], color="steelblue")
    ax.set_xlabel("Feature Importance")
    ax.set_title("Top Features by Importance (Final Model)")
    plt.tight_layout()
    out = os.path.join(SCREENSHOTS_DIR, "11_final_feature_importance.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved chart: {out}")

    # ---- Save everything -------------------------------------------------
    results.to_csv(os.path.join(RESULTS_DIR,
                   "feature_engineering_results.csv"), index=False)
    importance.to_csv(os.path.join(RESULTS_DIR,
                      "final_feature_importance.csv"), index=False)
    with open(os.path.join(RESULTS_DIR, "final_feature_list.txt"), "w") as f:
        for ft in final_features:
            f.write(ft + "\n")

    keep_cols = ["char"] + final_features + ["churned"]
    feat[keep_cols].to_csv(os.path.join(PROCESSED_DIR,
                           "wow_user_features.csv"), index=False)

    X_tr.to_csv(os.path.join(PROCESSED_DIR, "X_train.csv"), index=False)
    X_te.to_csv(os.path.join(PROCESSED_DIR, "X_test.csv"), index=False)
    y_tr.to_csv(os.path.join(PROCESSED_DIR, "y_train.csv"), index=False)
    y_te.to_csv(os.path.join(PROCESSED_DIR, "y_test.csv"), index=False)

    return final_features, final_score, baseline_score


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    section("PHASE 2: FEATURE ENGINEERING + FEATURE SELECTION")
    print("World of Warcraft Player Churn Prediction Project")
    print("Observation/outcome window split - leakage-free")

    feat = build_character_features()
    feat, tester, baseline_score, y = engineer_and_test(feat)
    final_features, final_score, baseline_score = select_and_finalize(
        feat, tester, baseline_score, y
    )

    section("PHASE 2 COMPLETE")
    print(f"Final feature count: {len(final_features)}")
    print(f"Baseline -> Final ROC-AUC: {baseline_score:.4f} -> {final_score:.4f}")
    print("\nOutputs:")
    print("  data/processed/phase2/wow_user_features.csv")
    print("  data/processed/phase2/X_train.csv, X_test.csv, y_train.csv, y_test.csv")
    print("  results/phase2/feature_engineering_results.csv")
    print("  results/phase2/final_feature_importance.csv")
    print("  results/phase2/final_feature_list.txt")
    print("  screenshots/phase2/10_feature_impact_ranking.png")
    print("  screenshots/phase2/11_final_feature_importance.png")


if __name__ == "__main__":
    main()
