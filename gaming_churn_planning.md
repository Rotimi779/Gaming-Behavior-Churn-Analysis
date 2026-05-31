# 🎮 WoW Player Churn Prediction — Project Plan

> **Reference document.** Master plan for the World of Warcraft player churn
> prediction project. Come back here any time to check scope, decisions, and
> next steps.
> Last updated: May 24, 2026

---

## Project Overview

**Project Name:** Player Churn Prediction — World of Warcraft

**Business Problem:** Online game publishers need to identify players who are
about to stop playing so they can intervene with targeted retention campaigns
(in-game rewards, personalised content, re-engagement messages) before the
player leaves permanently.

**Dataset:** World of Warcraft Avatar History (WoWAH) — a full year of
in-game observations from a real WoW server, sampled every 10 minutes across
the Horde faction.

- **Source:** Academic dataset released by Lee et al. (MMSys 2011)
- **File:** `wowah_data_filtered_10000.csv` (74 MB, hosted on GitHub)
- **Rows:** 1,380,068 observations
- **Characters:** 1,766 unique player characters
- **Date range:** 1 Jan 2008 – 31 Dec 2008 (365 days)

**Goal:** Aggregate raw snapshots into per-character features, train a binary
classifier to predict churn, and identify the strongest behavioural drivers.

**Code format:** All work done in `.py` scripts (not notebooks).

---

## Dataset — What's In It

Each row is a **10-minute snapshot** of one character being observed online.

| Column | Type | Description |
|--------|------|-------------|
| `char` | int | Anonymised character ID |
| `level` | int | Character level at time of snapshot (1–80) |
| `race` | str | One of: Orc, Tauren, Troll, Undead, Blood Elf |
| `charclass` | str | One of: Hunter, Warrior, Shaman, Warlock, Druid, Rogue, Priest, Mage, Paladin, Death Knight |
| `zone` | str | In-game zone (156 unique zones) |
| `guild` | int | Guild ID (-1 = no guild) |
| `timestamp` | str | Date + time of snapshot (MM/DD/YY HH:MM:SS) |

**Key facts from data inspection:**
- 5 races, 10 classes, 156 zones, levels 1–80
- 66% of characters (1,168 / 1,766) belong to a guild
- Median activity span: 200 days across the year
- Observations per character: very skewed (median 60, max 42,801)

---

## Target Variable — Churn Definition

A character is **churned** if they were **not seen for the final 60 days** of
the study period (i.e., last seen before 1 Nov 2008).

```
study_end = 2008-12-31
churned = (last_seen_timestamp < study_end - 60 days)
```

**Why 60 days?** Tested multiple cutoffs on the real data:

| Cutoff | Churned | Churn rate |
|--------|---------|-----------|
| 30 days | 1,076 | 60.9% — too high |
| 45 days | 938 | 53.1% — borderline |
| **60 days** | **838** | **47.5% — healthy range** |
| 90 days | 665 | 37.7% — also fine |

60 days is the most defensible business definition: "a player who has been
absent for 2 months is considered churned." It gives a healthy 47.5% churn
rate (within the ideal 15–55% range for ML).

**Verified signal on real data (60-day churn):**

| Feature | Churned mean | Retained mean | Correlation |
|---------|-------------|--------------|-------------|
| max_level | 38.8 | 57.2 | −0.337 |
| observations | 258 | 1,254 | −0.285 |
| activity_days | 93 | 265 | **−0.590** |
| zone_diversity | 9.2 | 34.9 | **−0.445** |
| has_guild | 0.5 | 0.8 | −0.269 |
| level_gain | 1.7 | 5.2 | −0.215 |

Real, strong correlations — completely opposite to the synthetic gaming
dataset that was rejected earlier.

---

## The Core Challenge: Snapshots → Character Features

Raw data is one row per **10-minute snapshot**. ML needs one row per
**character**. Phase 2 is entirely dedicated to this aggregation.

```
Raw (1,380,068 rows):
char  | level | race | charclass | zone     | guild | timestamp
59425 | 45    | Orc  | Hunter    | Barrens  | 165   | 01/15/08 14:02
59425 | 45    | Orc  | Hunter    | Barrens  | 165   | 01/15/08 14:12
59425 | 46    | Orc  | Hunter    | Thousand | 165   | 01/16/08 09:04
...

After aggregation (1,766 rows):
char  | max_level | observations | activity_days | zone_diversity | has_guild | ... | churned
59425 | 70        | 412          | 187           | 28             | 1         | ... | 0
```

---

## Project Structure

```
wow-churn/
├── data/
│   ├── raw/
│   │   └── wowah_data_filtered_10000.csv
│   └── processed/
│       ├── wow_user_features.csv          # after Phase 2
│       ├── X_train.csv / X_test.csv
│       └── y_train.csv / y_test.csv
├── src/
│   ├── phase1_eda.py                      # EDA on raw snapshots
│   ├── phase2_feature_engineering.py      # Aggregate snapshots → character features
│   ├── phase3_modeling.py                 # Train & compare models
│   └── phase4_evaluation_shap.py          # SHAP interpretation
├── models/
│   ├── best_model.pkl
│   ├── preprocessor.pkl
│   └── model_comparison_results.csv
├── results/
│   ├── eda_summary.txt
│   ├── feature_engineering_results.csv
│   ├── final_feature_importance.csv
│   ├── final_feature_list.txt
│   ├── business_insights.txt
│   └── shap_analysis/
├── app/
│   └── streamlit_app.py
├── screenshots/
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Timeline (2–3 Weeks)

| Phase | Days | Deliverable |
|-------|------|-------------|
| Phase 1: EDA on raw snapshots | 1–2 | `phase1_eda.py` |
| Phase 2: Aggregation + Feature Engineering | 2–3 | `phase2_feature_engineering.py` |
| Phase 3: Modeling | 2–3 | `phase3_modeling.py` |
| Phase 4: SHAP Evaluation | 1–2 | `phase4_evaluation_shap.py` |
| Phase 5: Dashboard | 2–3 | `streamlit_app.py` |
| Phase 6: Docs & Deployment | 1–2 | README + screenshots |

---

## PHASE 1 — EDA on Raw Snapshots

**Script:** `src/phase1_eda.py`

### Steps

1. **Load CSV** — read the raw file, strip column name whitespace, parse
   timestamps.
2. **Data quality** — missing values, timestamp parse failures, invalid levels,
   guild encoding (-1 = no guild).
3. **Define and verify churn label** — apply the 60-day rule; confirm 47.5%
   churn rate.
4. **Observation distribution** — how many snapshots per character (highly
   skewed); note the long tail.
5. **Activity span analysis** — days between first and last seen, split by
   churn status.
6. **Levelling patterns** — level distribution at first vs last snapshot;
   level gain over time; churned vs retained.
7. **Zone diversity** — unique zones visited per character, churned vs
   retained.
8. **Guild analysis** — guild membership rate, churn rate by guild vs no guild.
9. **Class and race breakdown** — churn rate by class and race.
10. **Temporal patterns** — when did characters go inactive? Were there waves
    of churn at specific months?
11. **Correlation analysis** — correlation of all engineerable raw features
    with churn; confirm the signals found in pre-inspection.
12. **Save** — `results/eda_summary.txt` + all charts.

### Phase 1 deliverables
- `results/eda_summary.txt`
- `screenshots/01_churn_distribution.png`
- `screenshots/02_activity_span_churned_vs_retained.png`
- `screenshots/03_level_distribution.png`
- `screenshots/04_zone_diversity.png`
- `screenshots/05_class_race_churn_rates.png`
- `screenshots/06_temporal_churn_pattern.png`
- `screenshots/07_correlation_with_churn.png`

---

## PHASE 2 — Feature Engineering + Rigorous Testing

**Script:** `src/phase2_feature_engineering.py`

This phase has three parts.

---

### Part A: Aggregate Snapshots → Character-Level Features

Group the 1,380,068 snapshot rows by `char` and compute the following
**baseline features** (direct aggregations — no engineering yet):

```python
# For each character compute:
observations          = total row count for this character
max_level             = highest level ever reached
min_level             = level at first observation
level_gain            = max_level - min_level
activity_days         = (last_seen - first_seen).days
days_since_start      = (first_seen - study_start).days   # how early they joined
unique_zones          = count of distinct zones visited
unique_sessions       = approximate sessions (gaps > 1 hour in timestamps)
has_guild             = 1 if guild != -1 ever, else 0
guild_count           = number of distinct guilds (guild changes = instability)
race_encoded          = label-encoded race
class_encoded         = label-encoded charclass
```

These are the **initial features** — what you can compute directly from the
raw columns before any creative engineering.

---

### Part B: Engineer & Test Additional Features

Use the `FeatureTester` framework — add one feature at a time, train a
Random Forest on baseline + new feature, measure ROC-AUC delta, KEEP if
delta > +0.001.

**Feature Group 1: Levelling Behaviour**
- `level_per_day` = max_level / (activity_days + 1) — levelling speed
- `is_max_level` = 1 if max_level == 70 or 80 (reached endgame; may have
  less reason to log in)
- `is_low_level` = 1 if max_level below 25th percentile
- `stalled_leveller` = level_gain < 2 AND activity_days > 30 (logged in but
  not progressing)
- `fast_leveller` = level_per_day above 75th percentile

**Feature Group 2: Activity Intensity**
- `obs_per_day` = observations / (activity_days + 1) — how frequently they
  played when active
- `is_heavy_player` = obs_per_day above 75th percentile
- `is_light_player` = obs_per_day below 25th percentile
- `early_dropout` = activity_days < 14 (quit within the first two weeks)
- `recency_ratio` = days_since_start / 365 (joined early vs late in the year)

**Feature Group 3: Zone Exploration**
- `zone_per_day` = unique_zones / (activity_days + 1) — exploration rate
- `is_explorer` = unique_zones above 75th percentile
- `is_static` = unique_zones below 25th percentile (rarely left one area)
- `zones_per_level` = unique_zones / (max_level + 1)

**Feature Group 4: Social / Guild Behaviour**
- `guild_stability` = 1 if guild_count == 1 else 0 (stayed in same guild)
- `guild_hopper` = 1 if guild_count > 2
- `no_guild` = 1 if has_guild == 0 (solo player — less socially anchored)
- `joined_guild_late` = guild membership started after first 30 days

**Feature Group 5: Interaction Features**
- `level_x_zones` = max_level * unique_zones
- `obs_x_level` = observations * max_level
- `activity_x_guild` = activity_days * has_guild

---

### Part C: Rank, Select, and Build Final Feature Set

1. Run `FeatureTester` on every engineered feature — print result per feature
   as it runs.
2. Print full ranked table sorted by ROC-AUC delta (descending).
3. Visualise: bar chart of all deltas, coloured KEEP (green) vs DISCARD (red).
4. Keep only features where delta > +0.001.
5. Retrain final model on baseline + kept features.
6. Report total ROC-AUC improvement over baseline.
7. Print final feature importance ranking.
8. Save train/test splits (stratified, 80/20) for Phase 3.

### Phase 2 deliverables
- `data/processed/wow_user_features.csv`
- `data/processed/X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`
- `results/feature_engineering_results.csv`
- `results/final_feature_importance.csv`
- `results/final_feature_list.txt`
- `screenshots/08_feature_impact_ranking.png`
- `screenshots/09_final_feature_importance.png`

---

## PHASE 3 — Modeling & Comparison

**Script:** `src/phase3_modeling.py`

### Steps

1. Load train/test splits from Phase 2.
2. Train four models with class-imbalance handling:
   - **Logistic Regression** (`class_weight='balanced'`) — interpretable
     baseline
   - **Random Forest** (`class_weight='balanced'`) — handles non-linear
     patterns
   - **XGBoost** (`scale_pos_weight`) — usually best performer
   - **LightGBM** (`class_weight='balanced'`) — fast, often matches XGBoost
3. Evaluate all models: ROC-AUC, Precision, Recall, F1-Score. Report together
   — never accuracy alone on imbalanced data.
4. Plot ROC curves for all four models on one chart.
5. Bar chart comparing all metrics across models.
6. Select best model by ROC-AUC.
7. Detailed evaluation of best model: classification report + confusion matrix
   with written interpretation.
8. Hyperparameter tuning with `GridSearchCV` (5-fold stratified CV, scored on
   ROC-AUC).
9. Save best model + all results.

**Note on sample size:** 1,766 characters is workable, but smaller than ideal
for ensemble methods. Use `StratifiedKFold` to preserve churn ratio in every
fold. Mention variance in the README.

### Phase 3 deliverables
- `models/best_model.pkl` (+ individual model pickles)
- `models/model_comparison_results.csv`
- `screenshots/10_roc_curves.png`
- `screenshots/11_model_comparison.png`
- `screenshots/12_confusion_matrix.png`

---

## PHASE 4 — Model Interpretation (SHAP)

**Script:** `src/phase4_evaluation_shap.py`

### Steps

1. Load best model + test data.
2. `shap.TreeExplainer` → compute SHAP values on the test set.
3. **Global importance** — SHAP bar plot (mean absolute SHAP per feature).
4. **SHAP beeswarm** — shows direction and magnitude for every test character.
5. **Individual character explanations** — waterfall plots for 3 highest-risk
   characters, showing exactly which features drove the prediction.
6. **Business insights** — translate the top SHAP drivers into plain-English
   retention strategies. Save to `results/business_insights.txt`.

### Expected top churn drivers (hypothesis, to be verified)
Based on pre-inspection correlations:
- `activity_days` — the strongest raw signal (−0.59); players who went quiet
  early are the biggest risk
- `zone_diversity` — players who explored more stayed longer (−0.45)
- `max_level` — higher level = more invested = less churn (−0.34)
- `has_guild` — social anchors keep players (−0.27)
- `obs_per_day` — light players churn more

### Phase 4 deliverables
- `results/shap_analysis/` (values + plots)
- `results/business_insights.txt`
- `screenshots/13_shap_bar.png`
- `screenshots/14_shap_beeswarm.png`
- `screenshots/15_shap_waterfall_sample.png`

---

## PHASE 5 — Streamlit Dashboard

**File:** `app/streamlit_app.py`

### Pages

**Page 1: Single Character Prediction**
Input a character's stats manually → get churn probability gauge + risk tier
(Low / Medium / High) + retention recommendations.

**Page 2: Batch Predictions**
Upload a CSV of aggregated character features → churn probabilities for all
characters → risk distribution chart → downloadable results with scores.

**Page 3: Model Performance**
Model comparison table, ROC curves, confusion matrix, SHAP importance chart.

**Page 4: About**
Dataset overview, methodology, key insights, business recommendations, link
to GitHub.

### Phase 5 deliverables
- Working Streamlit app (tested locally)
- Dashboard screenshots for README

---

## PHASE 6 — Documentation & Deployment

### README sections
- Project overview + business context
- Dataset description (what WoWAH is, why it's real data)
- Churn definition and why 60 days
- Results table (all four models)
- Key insights from SHAP (top churn drivers + retention strategies)
- Screenshots (all 15)
- Tech stack table
- Methodology (Phase 1–4 summary)
- How to run locally
- Contact

### Deployment (optional)
Streamlit Cloud — connect GitHub repo, set entry point to
`app/streamlit_app.py`.

---

## Evaluation Metrics — Quick Reference

| Metric | Question it answers | Notes |
|--------|---------------------|-------|
| **ROC-AUC** | How well does the model rank characters by churn risk? | Primary metric |
| **Precision** | Of predicted churners, how many actually churned? | Wasted retention spend |
| **Recall** | Of actual churners, how many did we catch? | Missed at-risk players |
| **F1-Score** | Balance of precision and recall | Single summary |
| **Confusion Matrix** | Exact breakdown of all four outcomes | Where does model fail? |
| **Accuracy** | % correct overall | **Never use alone — misleading** |

Metrics are always reported together. Business-driven threshold tuning is a
separate step applied after the best model is selected.

---

## Key Decisions Locked In

1. **Dataset:** WoWAH filtered CSV — 1,766 characters, 1 year, real signal.
2. **Churn definition:** Not seen for final 60 days of study = churned.
   Gives 47.5% churn rate, verified before committing.
3. **Feature selection:** Every engineered feature tested individually.
   KEEP threshold: ROC-AUC delta > +0.001. Features ranked by delta descending.
4. **Models:** Four trained and compared — Logistic Regression, Random Forest,
   XGBoost, LightGBM. Best selected by ROC-AUC.
5. **Code format:** `.py` scripts, not notebooks.
6. **Class imbalance:** Handled via `class_weight='balanced'` or
   `scale_pos_weight` depending on model. SMOTE as optional alternative.

---

## Why WoW — The Interview Talking Point

> "I used the World of Warcraft Avatar History dataset — a year of real
> player telemetry from an MMO server, sampled every 10 minutes. I chose it
> because it's one of the very few publicly available datasets with genuine
> per-player behavioural signals. I defined churn as a character absent for
> the final 60 days of the study — a 2-month inactivity window that mirrors
> how game publishers think about lapsed players. Pre-inspection confirmed
> strong correlations: activity span correlated at −0.59, zone diversity at
> −0.45 — meaning players who explored more of the game world genuinely
> stayed longer. That kind of natural signal is what makes the downstream
> model credible."

---

## Success Criteria

Portfolio-ready when:
1. ROC-AUC > 0.75 (adjusted for 1,766-character sample size)
2. At least 3 models compared (4 planned)
3. SHAP analysis showing top 10 churn drivers with business interpretation
4. Working Streamlit dashboard
5. Professional README with all screenshots
6. Clean, documented `.py` scripts
7. Concrete retention recommendations grounded in model insights

---

## Progress Tracker

- [ ] Phase 1: EDA on raw snapshots
- [ ] Phase 2: Feature engineering (aggregation + testing)
- [ ] Phase 3: Modeling
- [ ] Phase 4: SHAP
- [ ] Phase 5: Dashboard
- [ ] Phase 6: Documentation

*Last updated: May 24, 2026*
