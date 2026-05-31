"""
Phase 1: Exploratory Data Analysis  (Observation/Outcome Window Split)
World of Warcraft Player Churn Prediction Project

Churn is framed with an observation/outcome window split to prevent target
leakage:
  - Observation window: Jan 1 - Sep 30, 2008  -> ALL features come from here
  - Outcome window:      Oct 1 - Dec 31, 2008  -> churn label comes from here
  - churned = active in the observation window but ZERO snapshots in the
    outcome window.
  - Characters appearing ONLY in the outcome window (new joiners) are excluded.

This script:
  1. Loads the raw WoW snapshot data and splits it into the two windows
  2. Runs data quality checks
  3. Defines the churn label from the window split
  4. Analyzes observation volume, activity span, leveling, zones, guilds
     -- ALL computed from the observation window only
  5. Breaks down churn by class and race
  6. Examines temporal patterns within the observation window
  7. Computes feature correlations with churn
  8. Saves the labeled character summary, an EDA summary, and charts

Run from the project root:
    python src/phase1_eda.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Paths & config
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA = os.path.join(PROJECT_ROOT, "data", "raw",
                        "wowah_data_filtered_10000.csv")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
SCREENSHOTS_DIR = os.path.join(PROJECT_ROOT, "screenshots")

for d in (PROCESSED_DIR, RESULTS_DIR, SCREENSHOTS_DIR):
    os.makedirs(d, exist_ok=True)

sns.set_style("whitegrid")
RETAINED_COLOR = "#2ecc71"
CHURNED_COLOR = "#e74c3c"

# Window split boundaries
OBS_END = pd.Timestamp("2008-09-30 23:59:59")   # observation window ends here
OUTCOME_START = pd.Timestamp("2008-10-01")      # outcome window starts here


def section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# ---------------------------------------------------------------------------
# Step 1: Load & split into windows
# ---------------------------------------------------------------------------
def load_and_split():
    section("STEP 1: LOAD & SPLIT INTO OBSERVATION / OUTCOME WINDOWS")
    df = pd.read_csv(RAW_DATA)
    df.columns = [c.strip() for c in df.columns]
    df["timestamp"] = pd.to_datetime(
        df["timestamp"], format="%m/%d/%y %H:%M:%S", errors="coerce"
    )

    print(f"Raw dataset: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"Full date range: {df['timestamp'].min().date()} to "
          f"{df['timestamp'].max().date()}")

    df_obs = df[df["timestamp"] <= OBS_END].copy()
    df_outcome = df[df["timestamp"] >= OUTCOME_START].copy()

    print(f"\nObservation window (Jan 1 - Sep 30): {len(df_obs):,} rows")
    print(f"Outcome window     (Oct 1 - Dec 31): {len(df_outcome):,} rows")
    print(f"\nFirst 5 rows (observation window):")
    print(df_obs.head().to_string())

    return df, df_obs, df_outcome


# ---------------------------------------------------------------------------
# Step 2: Data quality check
# ---------------------------------------------------------------------------
def data_quality_check(df, df_obs):
    section("STEP 2: DATA QUALITY CHECK")

    missing = df.isnull().sum()
    print("Missing values per column:")
    print(missing.to_string())

    bad_ts = df["timestamp"].isnull().sum()
    print(f"\nTimestamp parse failures: {bad_ts}")

    dupes = df.duplicated().sum()
    print(f"Duplicate rows: {dupes:,}")

    print(f"\nObservation window summary:")
    print(f"  Level range: {df_obs['level'].min()} - {df_obs['level'].max()}")
    no_guild = (df_obs["guild"] == -1).sum()
    print(f"  Rows with guild = -1 (no guild): {no_guild:,} "
          f"({no_guild / len(df_obs) * 100:.1f}%)")
    print(f"  Races: {sorted(df_obs['race'].unique())}")
    print(f"  Classes: {sorted(df_obs['charclass'].unique())}")
    print(f"  Unique zones: {df_obs['zone'].nunique()}")
    print(f"  Unique characters: {df_obs['char'].nunique():,}")


# ---------------------------------------------------------------------------
# Step 3: Define churn label from the window split
# ---------------------------------------------------------------------------
def define_churn(df_obs, df_outcome):
    section("STEP 3: DEFINE CHURN LABEL (WINDOW SPLIT)")

    chars_obs = set(df_obs["char"].unique())
    chars_outcome = set(df_outcome["char"].unique())

    only_outcome = chars_outcome - chars_obs
    print(f"Characters in observation window: {len(chars_obs):,}")
    print(f"Characters in outcome window:     {len(chars_outcome):,}")
    print(f"Characters ONLY in outcome window: {len(only_outcome)} "
          f"-> EXCLUDED (new joiners with no observation-window history)")

    # Valid population = appeared in the observation window
    # churned = in observation window but NOT in outcome window
    churned_chars = chars_obs - chars_outcome

    churn_series = pd.Series(
        {c: (1 if c in churned_chars else 0) for c in chars_obs},
        name="churned"
    )
    churn_series.index.name = "char"

    counts = churn_series.value_counts()
    pct = churn_series.value_counts(normalize=True) * 100
    print(f"\nChurn label ({len(churn_series):,} valid characters):")
    print(f"  Retained (0): {counts.get(0, 0):>6,} ({pct.get(0, 0):5.2f}%)")
    print(f"  Churned  (1): {counts.get(1, 0):>6,} ({pct.get(1, 0):5.2f}%)")
    print("  (Note: rate shown is before the Phase 2 data-quality filters)")

    # Chart 01
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(["Retained", "Churned"], [counts.get(0, 0), counts.get(1, 0)],
                color=[RETAINED_COLOR, CHURNED_COLOR])
    axes[0].set_ylabel("Number of Characters")
    axes[0].set_title("Churn Distribution (Counts)")
    for i, v in enumerate([counts.get(0, 0), counts.get(1, 0)]):
        axes[0].text(i, v + 10, f"{v:,}", ha="center", fontweight="bold")

    axes[1].pie([counts.get(0, 0), counts.get(1, 0)],
                labels=["Retained", "Churned"], autopct="%1.1f%%",
                colors=[RETAINED_COLOR, CHURNED_COLOR], startangle=90)
    axes[1].set_title("Churn Rate (observation/outcome split)")

    plt.tight_layout()
    out = os.path.join(SCREENSHOTS_DIR, "01_churn_distribution.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved chart: {out}")

    return churn_series


# ---------------------------------------------------------------------------
# Build character summary FROM THE OBSERVATION WINDOW ONLY
# ---------------------------------------------------------------------------
def build_character_summary(df_obs, churn_series):
    """One row per character; all metrics from the observation window only."""
    obs_start = df_obs["timestamp"].min()
    g = df_obs.groupby("char")

    first_seen = g["timestamp"].min()
    last_seen = g["timestamp"].max()

    summary = pd.DataFrame({
        "observations": g.size(),
        "max_level": g["level"].max(),
        "min_level": g["level"].min(),
        "first_seen": first_seen,
        "last_seen": last_seen,
        "activity_days": (last_seen - first_seen).dt.days,
        "days_since_start": (first_seen - obs_start).dt.days,
        "unique_zones": g["zone"].nunique(),
        "guild_count": g["guild"].apply(lambda s: s[s != -1].nunique()),
        "race": g["race"].first(),
        "charclass": g["charclass"].first(),
    })
    summary["level_gain"] = summary["max_level"] - summary["min_level"]
    summary["has_guild"] = (summary["guild_count"] > 0).astype(int)
    summary["churned"] = churn_series
    return summary


# ---------------------------------------------------------------------------
# Step 4: Observation distribution
# ---------------------------------------------------------------------------
def observation_distribution(summary):
    section("STEP 4: OBSERVATION DISTRIBUTION (observation window)")

    obs = summary["observations"]
    print("Observations per character (Jan-Sep):")
    print(obs.describe().round(1).to_string())

    few = (obs < 5).sum()
    print(f"\nCharacters with < 5 observations: {few} "
          f"({few / len(summary) * 100:.1f}%) "
          f"-- will be filtered in Phase 2")

    fig, ax = plt.subplots(figsize=(12, 6))
    bins = np.logspace(0, np.log10(obs.max() + 1), 40)
    ax.hist([summary[summary.churned == 0]["observations"],
             summary[summary.churned == 1]["observations"]],
            bins=bins, label=["Retained", "Churned"],
            color=[RETAINED_COLOR, CHURNED_COLOR])
    ax.set_xscale("log")
    ax.set_xlabel("Observations per Character (log scale)")
    ax.set_ylabel("Number of Characters")
    ax.set_title("Observation Volume: Churned vs Retained (Jan-Sep)")
    ax.legend()
    plt.tight_layout()
    out = os.path.join(SCREENSHOTS_DIR, "02_observations_per_character.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved chart: {out}")


# ---------------------------------------------------------------------------
# Step 5: Activity span analysis
# ---------------------------------------------------------------------------
def activity_span_analysis(summary):
    section("STEP 5: ACTIVITY SPAN ANALYSIS (observation window)")

    by_churn = summary.groupby("churned")["activity_days"].agg(
        ["mean", "median", "std"]).round(1)
    print("Activity span in days (first to last seen, Jan-Sep):")
    print(by_churn.to_string())

    retained_mean = summary[summary.churned == 0]["activity_days"].mean()
    churned_mean = summary[summary.churned == 1]["activity_days"].mean()
    print(f"\nWithin the observation window, retained players are active for "
          f"{retained_mean:.0f} days on average vs {churned_mean:.0f} for "
          f"churned players.")

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(data=summary, x="churned", y="activity_days", ax=ax,
                hue="churned", palette=[RETAINED_COLOR, CHURNED_COLOR],
                legend=False)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Retained", "Churned"])
    ax.set_xlabel("")
    ax.set_ylabel("Activity Span (days, Jan-Sep)")
    ax.set_title("Activity Span: Churned vs Retained")
    plt.tight_layout()
    out = os.path.join(SCREENSHOTS_DIR, "03_activity_span.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved chart: {out}")


# ---------------------------------------------------------------------------
# Step 6: Leveling patterns
# ---------------------------------------------------------------------------
def leveling_patterns(summary):
    section("STEP 6: LEVELING PATTERNS (observation window)")

    by_churn = summary.groupby("churned")[["max_level", "level_gain"]].agg(
        ["mean", "median"]).round(1)
    print("Max level and level gain by churn status (by Sep 30):")
    print(by_churn.to_string())

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].hist([summary[summary.churned == 0]["max_level"],
                  summary[summary.churned == 1]["max_level"]],
                 bins=40, label=["Retained", "Churned"],
                 color=[RETAINED_COLOR, CHURNED_COLOR])
    axes[0].set_xlabel("Max Level Reached (by Sep 30)")
    axes[0].set_ylabel("Number of Characters")
    axes[0].set_title("Max Level: Churned vs Retained")
    axes[0].legend()

    sns.boxplot(data=summary, x="churned", y="level_gain", ax=axes[1],
                hue="churned", palette=[RETAINED_COLOR, CHURNED_COLOR],
                legend=False)
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(["Retained", "Churned"])
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Level Gain (max - min)")
    axes[1].set_title("Level Progression: Churned vs Retained")
    plt.tight_layout()
    out = os.path.join(SCREENSHOTS_DIR, "04_level_distribution.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved chart: {out}")


# ---------------------------------------------------------------------------
# Step 7: Zone diversity analysis
# ---------------------------------------------------------------------------
def zone_diversity_analysis(df_obs, summary):
    section("STEP 7: ZONE DIVERSITY ANALYSIS (observation window)")

    by_churn = summary.groupby("churned")["unique_zones"].agg(
        ["mean", "median"]).round(1)
    print("Unique zones visited by churn status (Jan-Sep):")
    print(by_churn.to_string())

    top_zones = df_obs["zone"].value_counts().head(10)
    print("\nTop 10 most-visited zones (observation window):")
    for zone, cnt in top_zones.items():
        print(f"  {zone:<28} {cnt:>9,}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.boxplot(data=summary, x="churned", y="unique_zones", ax=axes[0],
                hue="churned", palette=[RETAINED_COLOR, CHURNED_COLOR],
                legend=False)
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(["Retained", "Churned"])
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Unique Zones Visited (Jan-Sep)")
    axes[0].set_title("Zone Diversity: Churned vs Retained")

    axes[1].barh(top_zones.index[::-1], top_zones.values[::-1],
                 color="steelblue")
    axes[1].set_xlabel("Snapshot Count")
    axes[1].set_title("Top 10 Most-Visited Zones (Jan-Sep)")
    plt.tight_layout()
    out = os.path.join(SCREENSHOTS_DIR, "05_zone_diversity.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved chart: {out}")


# ---------------------------------------------------------------------------
# Step 8: Guild analysis
# ---------------------------------------------------------------------------
def guild_analysis(summary):
    section("STEP 8: GUILD ANALYSIS (observation window)")

    guild_rate = summary["has_guild"].mean() * 100
    print(f"Characters ever in a guild (Jan-Sep): {guild_rate:.1f}%")

    churn_by_guild = summary.groupby("has_guild")["churned"].mean() * 100
    print("\nChurn rate by guild membership:")
    print(f"  No guild (solo):  {churn_by_guild.get(0, 0):.1f}%")
    print(f"  Guild member:     {churn_by_guild.get(1, 0):.1f}%")

    members = summary[summary["has_guild"] == 1]
    stable = (members["guild_count"] == 1).sum()
    hoppers = (members["guild_count"] > 1).sum()
    print(f"\nAmong guild members:")
    print(f"  Stayed in one guild: {stable:,}")
    print(f"  Switched guilds:     {hoppers:,}")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    axes[0].bar(["No Guild", "Guild Member"],
                [churn_by_guild.get(0, 0), churn_by_guild.get(1, 0)],
                color=[CHURNED_COLOR, RETAINED_COLOR])
    axes[0].set_ylabel("Churn Rate (%)")
    axes[0].set_title("Churn Rate by Guild Membership")
    for i, v in enumerate([churn_by_guild.get(0, 0), churn_by_guild.get(1, 0)]):
        axes[0].text(i, v + 1, f"{v:.1f}%", ha="center", fontweight="bold")

    axes[1].bar(["One Guild", "Switched Guilds"], [stable, hoppers],
                color="steelblue")
    axes[1].set_ylabel("Number of Characters")
    axes[1].set_title("Guild Stability (members only)")
    for i, v in enumerate([stable, hoppers]):
        axes[1].text(i, v + 3, f"{v:,}", ha="center", fontweight="bold")
    plt.tight_layout()
    out = os.path.join(SCREENSHOTS_DIR, "06_guild_analysis.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved chart: {out}")


# ---------------------------------------------------------------------------
# Step 9: Class & race breakdown
# ---------------------------------------------------------------------------
def class_race_breakdown(summary):
    section("STEP 9: CLASS & RACE BREAKDOWN (observation window)")

    churn_by_class = (summary.groupby("charclass")["churned"].mean() * 100
                      ).sort_values(ascending=False)
    churn_by_race = (summary.groupby("race")["churned"].mean() * 100
                     ).sort_values(ascending=False)

    print("Churn rate by class:")
    for cls, rate in churn_by_class.items():
        print(f"  {cls:<14} {rate:5.1f}%")
    print("\nChurn rate by race:")
    for race, rate in churn_by_race.items():
        print(f"  {race:<14} {rate:5.1f}%")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].barh(churn_by_class.index[::-1], churn_by_class.values[::-1],
                 color="coral")
    axes[0].set_xlabel("Churn Rate (%)")
    axes[0].set_title("Churn Rate by Class")

    axes[1].barh(churn_by_race.index[::-1], churn_by_race.values[::-1],
                 color="mediumpurple")
    axes[1].set_xlabel("Churn Rate (%)")
    axes[1].set_title("Churn Rate by Race")
    plt.tight_layout()
    out = os.path.join(SCREENSHOTS_DIR, "07_class_race_churn.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved chart: {out}")


# ---------------------------------------------------------------------------
# Step 10: Temporal pattern within the observation window
# ---------------------------------------------------------------------------
def temporal_pattern(summary):
    section("STEP 10: TEMPORAL PATTERN (within observation window)")

    churned = summary[summary["churned"] == 1].copy()
    churned["last_seen_month"] = churned["last_seen"].dt.to_period("M")
    monthly = churned.groupby("last_seen_month").size()

    print("Last-seen month of churned characters (within Jan-Sep window):")
    for month, cnt in monthly.items():
        print(f"  {month}  {cnt:>4}")
    print("\nNote: churned characters were last seen at various points in the")
    print("observation window; none appear after Sep 30 by definition.")

    fig, ax = plt.subplots(figsize=(12, 6))
    x = [str(m) for m in monthly.index]
    ax.plot(x, monthly.values, marker="o", color=CHURNED_COLOR, linewidth=2)
    ax.set_xlabel("Month (last seen within observation window)")
    ax.set_ylabel("Number of Churned Characters")
    ax.set_title("Last-Seen Month of Churned Characters (Jan-Sep)")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = os.path.join(SCREENSHOTS_DIR, "08_temporal_churn.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved chart: {out}")


# ---------------------------------------------------------------------------
# Step 11: Correlation summary
# ---------------------------------------------------------------------------
def correlation_summary(summary):
    section("STEP 11: CORRELATION SUMMARY")

    feats = ["activity_days", "unique_zones", "max_level", "observations",
             "has_guild", "level_gain", "guild_count", "days_since_start"]
    corr = summary[feats + ["churned"]].corr()["churned"].drop("churned")
    corr = corr.sort_values()

    print("Correlation of each observation-window metric with churn:")
    for feat, c in corr.items():
        print(f"  {feat:<20} {c:+.4f}")

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [CHURNED_COLOR if v > 0 else RETAINED_COLOR for v in corr.values]
    ax.barh(corr.index, corr.values, color=colors)
    ax.axvline(x=0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Correlation with Churn")
    ax.set_title("Observation-Window Feature Correlation with Churn "
                 "(green = protective, red = risk)")
    plt.tight_layout()
    out = os.path.join(SCREENSHOTS_DIR, "09_correlation_summary.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved chart: {out}")

    return corr


# ---------------------------------------------------------------------------
# Step 12: Save outputs
# ---------------------------------------------------------------------------
def save_outputs(summary, corr):
    section("STEP 12: SAVE OUTPUTS")

    data_out = os.path.join(PROCESSED_DIR, "wow_eda_character_summary.csv")
    summary.reset_index().to_csv(data_out, index=False)
    print(f"  Saved: {data_out}")

    summary_out = os.path.join(RESULTS_DIR, "eda_summary.txt")
    churn_rate = summary["churned"].mean() * 100
    with open(summary_out, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("PHASE 1: EDA SUMMARY - WoW Player Churn\n")
        f.write("Observation/Outcome Window Split\n")
        f.write("=" * 80 + "\n\n")
        f.write("Observation window: 2008-01-01 to 2008-09-30 "
                "(features come from here)\n")
        f.write("Outcome window:     2008-10-01 to 2008-12-31 "
                "(churn label comes from here)\n\n")
        f.write("Churn definition: active in observation window but ZERO\n")
        f.write("snapshots in the outcome window.\n\n")
        f.write(f"Valid characters (pre Phase-2 filters): {len(summary):,}\n")
        f.write(f"Churn rate: {churn_rate:.2f}%\n")
        f.write(f"Churned: {int(summary['churned'].sum()):,}  |  "
                f"Retained: {int((summary['churned'] == 0).sum()):,}\n\n")

        f.write("Mean metrics: churned vs retained (observation window)\n")
        for col in ["activity_days", "unique_zones", "max_level",
                    "observations", "level_gain"]:
            r = summary[summary.churned == 0][col].mean()
            c = summary[summary.churned == 1][col].mean()
            f.write(f"  {col:<18} retained={r:>9.1f}  churned={c:>9.1f}\n")

        f.write("\nFeature correlation with churn:\n")
        for feat, c in corr.items():
            f.write(f"  {feat:<20} {c:+.4f}\n")

        f.write("\nNOTE for README: the observation/outcome window split is a\n")
        f.write("deliberate choice to prevent target leakage. An earlier\n")
        f.write("full-year definition leaked last_seen into activity_days.\n")
    print(f"  Saved: {summary_out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    section("PHASE 1: EXPLORATORY DATA ANALYSIS (WINDOW SPLIT)")
    print("World of Warcraft Player Churn Prediction Project")

    df, df_obs, df_outcome = load_and_split()
    data_quality_check(df, df_obs)
    churn_series = define_churn(df_obs, df_outcome)

    summary = build_character_summary(df_obs, churn_series)

    observation_distribution(summary)
    activity_span_analysis(summary)
    leveling_patterns(summary)
    zone_diversity_analysis(df_obs, summary)
    guild_analysis(summary)
    class_race_breakdown(summary)
    temporal_pattern(summary)
    corr = correlation_summary(summary)
    save_outputs(summary, corr)

    section("PHASE 1 COMPLETE")
    print("Outputs:")
    print("  data/processed/wow_eda_character_summary.csv")
    print("  results/eda_summary.txt")
    print("  screenshots/01_churn_distribution.png")
    print("  screenshots/02_observations_per_character.png")
    print("  screenshots/03_activity_span.png")
    print("  screenshots/04_level_distribution.png")
    print("  screenshots/05_zone_diversity.png")
    print("  screenshots/06_guild_analysis.png")
    print("  screenshots/07_class_race_churn.png")
    print("  screenshots/08_temporal_churn.png")
    print("  screenshots/09_correlation_summary.png")


if __name__ == "__main__":
    main()
