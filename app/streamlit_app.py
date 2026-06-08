"""
WoW Player Churn Prediction - Streamlit Dashboard
Phase 5

An interactive dashboard for the World of Warcraft player churn model.

Pages:
  1. Single Character Prediction - enter a character's stats, get churn risk
  2. Batch Predictions        - upload a CSV of characters, score them all
  3. Model Performance        - comparison table, ROC curves, SHAP charts
  4. About                    - project overview and methodology

The user enters only the RAW behavioural inputs a game team would actually
have (observations, level, zones, sessions, guild, etc.). The app derives the
full 29-feature vector the model expects, scales it, and predicts.

Run locally:
    streamlit run app/streamlit_app.py
"""

import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st


import sys
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu

# ---------------------------------------------------------------------------
# Paths (relative to project root so it works locally and on Streamlit Cloud)
# ---------------------------------------------------------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(APP_DIR)
MODELS_DIR = os.path.join(ROOT, "models")

# WoW race / class label-encoding maps (alphabetical, matching LabelEncoder)
RACES = ["Blood Elf", "Orc", "Tauren", "Troll", "Undead"]
CLASSES = ["Druid", "Hunter", "Mage", "Paladin", "Priest",
           "Rogue", "Shaman", "Warlock", "Warrior"]

# The exact feature order the model was trained on
FEATURE_ORDER = [
    "observations", "max_level", "min_level", "activity_days",
    "days_since_start", "unique_zones", "unique_sessions", "guild_count",
    "level_gain", "has_guild", "race_encoded", "class_encoded",
    "obs_per_session", "zone_per_day", "level_per_day", "zones_per_obs",
    "is_late_joiner", "no_guild", "is_heavy_player", "is_light_player",
    "is_max_level", "is_static", "stalled_leveller", "is_low_level",
    "progression_ratio", "guild_hopper", "is_explorer", "obs_x_level",
    "activity_x_guild",
]


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
def load_model():
    model = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    return model, scaler


@st.cache_data
def load_reference_data():
    """Load the training feature table to derive percentile thresholds."""
    df = pd.read_csv(os.path.join(ROOT, "data", "processed", "phase2",
                                  "wow_user_features.csv"))
    return df


@st.cache_data
def load_comparison():
    path = os.path.join(ROOT, "results", "phase3",
                        "model_comparison_results.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


# ---------------------------------------------------------------------------
# Feature engineering - turn raw inputs into the 29-feature vector
# ---------------------------------------------------------------------------
def derive_thresholds(ref):
    """Compute the same percentile cut-offs Phase 2 used."""
    return {
        "opd_q75": (ref["observations"] / (ref["activity_days"] + 1)).quantile(0.75),
        "opd_q25": (ref["observations"] / (ref["activity_days"] + 1)).quantile(0.25),
        "zones_q75": ref["unique_zones"].quantile(0.75),
        "zones_q25": ref["unique_zones"].quantile(0.25),
        "level_q25": ref["max_level"].quantile(0.25),
    }


def build_feature_vector(raw, thresholds):
    """raw is a dict of the raw inputs; returns a single-row DataFrame."""
    observations = raw["observations"]
    max_level = raw["max_level"]
    min_level = raw["min_level"]
    activity_days = raw["activity_days"]
    days_since_start = raw["days_since_start"]
    unique_zones = raw["unique_zones"]
    unique_sessions = raw["unique_sessions"]
    guild_count = raw["guild_count"]

    level_gain = max_level - min_level
    has_guild = 1 if guild_count > 0 else 0

    obs_per_day = observations / (activity_days + 1)
    obs_per_session = observations / (unique_sessions + 1)
    zone_per_day = unique_zones / (activity_days + 1)
    level_per_day = max_level / (activity_days + 1)
    zones_per_obs = unique_zones / (observations + 1)
    progression_ratio = level_gain / (max_level + 1)

    feat = {
        "observations": observations,
        "max_level": max_level,
        "min_level": min_level,
        "activity_days": activity_days,
        "days_since_start": days_since_start,
        "unique_zones": unique_zones,
        "unique_sessions": unique_sessions,
        "guild_count": guild_count,
        "level_gain": level_gain,
        "has_guild": has_guild,
        "race_encoded": raw["race_encoded"],
        "class_encoded": raw["class_encoded"],
        "obs_per_session": obs_per_session,
        "zone_per_day": zone_per_day,
        "level_per_day": level_per_day,
        "zones_per_obs": zones_per_obs,
        "is_late_joiner": 1 if days_since_start > 180 else 0,
        "no_guild": 1 if has_guild == 0 else 0,
        "is_heavy_player": 1 if obs_per_day > thresholds["opd_q75"] else 0,
        "is_light_player": 1 if obs_per_day < thresholds["opd_q25"] else 0,
        "is_max_level": 1 if max_level >= 70 else 0,
        "is_static": 1 if unique_zones < thresholds["zones_q25"] else 0,
        "stalled_leveller": 1 if (level_gain < 2 and activity_days > 30) else 0,
        "is_low_level": 1 if max_level < thresholds["level_q25"] else 0,
        "progression_ratio": progression_ratio,
        "guild_hopper": 1 if guild_count > 2 else 0,
        "is_explorer": 1 if unique_zones > thresholds["zones_q75"] else 0,
        "obs_x_level": observations * max_level,
        "activity_x_guild": activity_days * has_guild,
    }
    return pd.DataFrame([feat])[FEATURE_ORDER]


def predict_churn(model, scaler, X):
    # Preserve column names through scaling so the model gets a named DataFrame
    X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)
    proba = model.predict_proba(X_scaled)[:, 1]
    return proba


def risk_tier(p):
    if p >= 0.66:
        return "High", "#e74c3c"
    if p >= 0.40:
        return "Medium", "#e67e22"
    return "Low", "#2ecc71"


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
def page_dashboard():

    st.title("🎮 World of Warcraft Churn Analytics")

    st.markdown("""
    Predict player retention using machine learning and behavioral telemetry.
    This dashboard helps identify at-risk players before they disengage.
    """)

    st.divider()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "ROC-AUC",
            "0.833"
        )

    with col2:
        st.metric(
            "Characters",
            "1,195"
        )

    with col3:
        st.metric(
            "Churn Rate",
            "31.5%"
        )

    with col4:
        st.metric(
            "Features",
            "29"
        )

    st.divider()

    left, right = st.columns([2,1])

    with left:

        st.subheader("Project Overview")

        st.markdown("""
        This machine learning system predicts whether a World of Warcraft
        character is likely to stop playing based on gameplay behavior
        observed during a nine-month observation period.

        The model was trained on aggregated behavioral telemetry derived
        from over one million gameplay observations.
        """)

        st.subheader("Key Churn Drivers")

        st.markdown("""
        <div class="insight-card">
        📉 Short activity spans strongly increase churn probability.
        </div>

        <div class="insight-card">
        🧭 Players who explore fewer zones are more likely to disengage.
        </div>

        <div class="insight-card">
        👥 Guild membership significantly improves retention.
        </div>

        <div class="insight-card">
        ⚔️ Progression and level advancement correlate with long-term engagement.
        </div>
        """, unsafe_allow_html=True)

    with right:

        st.subheader("Business Impact")

        st.info("""
        Early churn detection allows:

        • Targeted retention campaigns

        • Personalized content recommendations

        • Guild onboarding initiatives

        • Reduced player attrition
        """)

        st.success("""
        Best Model

        Logistic Regression

        ROC-AUC = 0.833
        """)

def risk_gauge(probability):

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%"},
            title={"text": "Churn Probability"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"thickness": 0.3},
                "steps": [
                    {"range": [0, 33], "color": "#dcfce7"},
                    {"range": [33, 66], "color": "#fef3c7"},
                    {"range": [66, 100], "color": "#fee2e2"},
                ],
            },
        )
    )

    fig.update_layout(height=300)

    return fig

def show_risk_card(probability, tier):

    if tier == "Low":

        st.success(
            f"""
            🟢 LOW CHURN RISK

            Estimated Probability:
            {probability:.1%}
            """
        )

    elif tier == "Medium":

        st.warning(
            f"""
            🟠 MEDIUM CHURN RISK

            Estimated Probability:
            {probability:.1%}
            """
        )

    else:

        st.error(
            f"""
            🔴 HIGH CHURN RISK

            Estimated Probability:
            {probability:.1%}
            """
        )



def page_single(model, scaler, ref, thresholds):
    st.title("🔮 Player Churn Prediction")

    st.caption(
        "Evaluate a player's risk of churn using gameplay behavior indicators."
    )

    st.divider()

    st.subheader("Gameplay Inputs")
    st.info(
        "Enter observed gameplay behaviour from the 9-month observation window."
    )
    # st.write("Enter a character's behaviour from the observation window "
    #          "(their activity over a 9-month period) to estimate churn risk.")

    col1, col2 = st.columns(2)
    with col1:
        observations = st.number_input(
            "Observed Activity Snapshots",
            min_value=5, max_value=40000, value=300, step=10)
        max_level = st.slider("Max level reached", 1, 70, 45)
        min_level = st.slider("Starting level (first seen)", 1, max_level, 1)
        activity_days = st.number_input(
            "Activity Duration (Days)",
            min_value=0, max_value=273, value=120, step=1)
        days_since_start = st.number_input(
            "First Appearance Day",
            min_value=0, max_value=273, value=10, step=1)
    with col2:
        unique_zones = st.slider("Unique zones visited", 1, 130, 20)
        unique_sessions = st.number_input(
            "Number of play sessions", min_value=1, max_value=2000,
            value=40, step=1)
        guild_count = st.slider("Number of guilds belonged to", 0, 5, 1)
        race = st.selectbox("Race", RACES, index=1)
        charclass = st.selectbox("Class", CLASSES, index=1)

    if st.button("Predict churn risk", type="primary"):
        raw = {
            "observations": observations, "max_level": max_level,
            "min_level": min_level, "activity_days": activity_days,
            "days_since_start": days_since_start, "unique_zones": unique_zones,
            "unique_sessions": unique_sessions, "guild_count": guild_count,
            "race_encoded": RACES.index(race),
            "class_encoded": CLASSES.index(charclass),
        }
        X = build_feature_vector(raw, thresholds)
        p = predict_churn(model, scaler, X)[0]
        tier, color = risk_tier(p)

        show_risk_card(p, tier)
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Risk Tier",
                tier
            )

        with col2:
            st.metric(
                "Probability",
                f"{p*100:.1f}%"
            )

        with col3:
            st.metric(
                "Guilds",
                guild_count
            )
        left, right = st.columns([1, 1])

        with left:

            st.plotly_chart(
                risk_gauge(p),
                use_container_width=True
            )
            st.caption(
                f"Risk Score: {p:.1%} | Generated using the production Logistic Regression model."
            )

        with right:

            st.subheader("Recommended Actions")

            recs = []

            if guild_count == 0:
                recs.append(
                    "Recommend guild onboarding to strengthen social retention."
                )

            if unique_zones < thresholds["zones_q25"]:
                recs.append(
                    "Encourage exploration through new zone recommendations."
                )

            if max_level < thresholds["level_q25"]:
                recs.append(
                    "Target player with progression-focused incentives."
                )

            if activity_days < 30:
                recs.append(
                    "Trigger re-engagement campaigns early."
                )

            if not recs:
                recs.append(
                    "Player demonstrates healthy engagement patterns."
                )

            for rec in recs:
                st.markdown(f"✅ {rec}")

        st.divider()

        st.subheader("Behavioral Interpretation")

        if tier == "High":

            st.error("""
            This player exhibits multiple behavioural signals
            associated with future churn and disengagement.

            Immediate retention actions are recommended.
            """)

        elif tier == "Medium":

            st.warning("""
            This player shows moderate churn risk.

            Additional engagement opportunities may improve
            long-term retention.
            """)

        else:

            st.success("""
            This player currently demonstrates healthy
            engagement patterns and low churn risk.
            """)


def page_batch(model, scaler, ref, thresholds):
    st.title("Batch Analytics")
    st.caption(
        "Upload a dataset of player behaviour records and generate churn predictions at scale."
    )

    st.download_button(
        "Download a template CSV",
        data=pd.DataFrame([{
            "observations": 300, "max_level": 45, "min_level": 1,
            "activity_days": 120, "days_since_start": 10, "unique_zones": 20,
            "unique_sessions": 40, "guild_count": 1, "race_encoded": 1,
            "class_encoded": 1,
        }]).to_csv(index=False),
        file_name="batch_template.csv", mime="text/csv")

    uploaded = st.file_uploader("Upload character CSV", type="csv")
    if uploaded is not None:
        df_in = pd.read_csv(uploaded)
        rows = []
        for _, r in df_in.iterrows():
            rows.append(build_feature_vector(r.to_dict(), thresholds))
        X = pd.concat(rows, ignore_index=True)
        probs = predict_churn(model, scaler, X)

        out = df_in.copy()
        out["churn_probability"] = probs.round(3)
        out["risk_tier"] = [risk_tier(p)[0] for p in probs]
        tab1, tab2 = st.tabs(
            [
                "📈 Analytics",
                "📋 Dataset"
            ]
        )
        with tab1:


            # ==========================
            # KPI Summary
            # ==========================

            total_players = len(out)

            high_risk = len(
                out[out["risk_tier"] == "High"]
            )

            avg_risk = out["churn_probability"].mean()

            high_risk_pct = (
                high_risk / total_players
            ) * 100

            st.subheader("Analytics Summary")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Players",
                    f"{total_players:,}"
                )

            with col2:
                st.metric(
                    "High Risk",
                    high_risk
                )

            with col3:
                st.metric(
                    "Average Risk",
                    f"{avg_risk:.1%}"
                )

            with col4:
                st.metric(
                    "High Risk %",
                    f"{high_risk_pct:.1f}%"
                )

            st.info(
                f"""
                {high_risk} of {total_players} uploaded players
                were classified as High Risk.

                The average churn probability across the cohort
                is {avg_risk:.1%}.
                """
            )

            st.subheader("Risk Tier Distribution")

            risk_counts = (
                out["risk_tier"]
                .value_counts()
                .reset_index()
            )

            risk_counts.columns = [
                "risk_tier",
                "count"
            ]

            fig = px.pie(
                risk_counts,
                names="risk_tier",
                values="count",
                title="Player Risk Distribution",
                hole=0.45
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.divider()

            st.subheader("🚨 Highest Risk Players")

            high_risk_df = (
                out[out["risk_tier"] == "High"]
                .sort_values(
                    "churn_probability",
                    ascending=False
                )
            )

            if len(high_risk_df) > 0:

                st.caption(
                    "Players most likely to churn and therefore strongest candidates for retention interventions."
                )

                st.dataframe(
                    high_risk_df.head(25),
                    use_container_width=True
                )

            else:

                st.success(
                    "No players were classified as High Risk."
                )

        with tab2:
            st.subheader("📋 Full Scored Dataset")
            st.dataframe(out.sort_values("churn_probability", ascending=False),
                        use_container_width=True)

            st.download_button(
                "Download scored results",
                data=out.to_csv(index=False),
                file_name="churn_predictions.csv", mime="text/csv")


def page_performance():
    st.header("Model performance")

    st.caption(
        "Evaluation of machine learning models used to predict player churn."
    )

    st.divider()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Best Model",
            "Logistic Regression"
        )

    with col2:
        st.metric(
            "ROC-AUC",
            "0.833"
        )

    with col3:
        st.metric(
            "Characters",
            "1,195"
        )

    with col4:
        st.metric(
            "Features",
            "17"
        )

    st.divider()

    tab1, tab2, tab3 = st.tabs(
        [
            "📊 Model Comparison",
            "📈 Evaluation",
            "🧠 SHAP Analysis"
        ]
    )

    with tab1:

        comparison = load_comparison()

        if comparison is not None:

            st.subheader("Model Comparison")

            st.dataframe(
                comparison.round(4),
                use_container_width=True
            )

            st.success(
                "Logistic Regression achieved the highest test ROC-AUC and was selected as the production model."
            )

    def show(phase, img, caption):
        path = os.path.join(ROOT, "screenshots", phase, img)
        if os.path.exists(path):
            st.image(path, caption=caption, use_container_width=True)


    with tab2:

        st.subheader("ROC Curves")

        show(
            "phase3",
            "12_roc_curves.png",
            "ROC curves for all four models"
        )

        st.subheader("Confusion Matrix")

        show(
            "phase3",
            "14_confusion_matrix.png",
            "Best model confusion matrix"
        )
    with tab3:

        st.subheader("Feature Importance")

        show(
            "phase4",
            "15_shap_bar.png",
            "Global SHAP feature importance"
        )

        show(
            "phase4",
            "16_shap_beeswarm.png",
            "SHAP beeswarm plot"
        )

    st.divider()

    st.subheader("Key Findings")

    st.markdown("""
    <div class="insight-card">
    ⚔️ Players with shorter activity spans are substantially more likely to churn.
    </div>

    <div class="insight-card">
    🗺️ Exploration diversity strongly correlates with retention.
    </div>

    <div class="insight-card">
    👥 Guild membership is one of the strongest protective factors against churn.
    </div>

    <div class="insight-card">
    📈 Character progression is associated with long-term engagement.
    </div>
    """, unsafe_allow_html=True)

def page_about():
    st.title("📘 About the Project")

    st.caption(
        "An end-to-end machine learning project for predicting player churn in World of Warcraft."
    )

    st.divider()
    st.subheader("🎮 Project Overview")

    st.markdown("""
    This dashboard predicts whether a World of Warcraft character will churn
    (stop playing) based on behavioural patterns observed during gameplay.

    The project uses real player telemetry and applies machine learning to identify
    players most at risk of leaving the game.
    """)

    st.subheader("📊 Dataset")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Snapshots",
            "1.38M"
        )

    with col2:
        st.metric(
            "Characters",
            "1,766"
        )

    with col3:
        st.metric(
            "Final Dataset",
            "1,195"
        )

    st.info(
        """
        Data comes from the WoW Avatar History dataset and consists of
        player activity snapshots collected every 10 minutes throughout 2008.
        """
    )

    st.subheader("⚠️ Churn Definition")

    st.markdown("""
    The year was split into two periods:

    - **Observation Window:** January–September
    - **Outcome Window:** October–December

    Features were created exclusively from the observation window.

    A player was classified as churned if they were active during the observation
    period but recorded no activity during the outcome period.
    """)

    st.subheader("🛠 Methodology")

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "EDA",
            "Features",
            "Modeling",
            "SHAP"
        ]
    )
    with tab1:

        st.markdown("""
        - Explored 1,560 characters
        - Investigated churn patterns
        - Identified behavioural indicators
        - Examined engagement distributions
        """)

    with tab2:

        st.markdown("""
        - Engineered 23 candidate features
        - Retained 17 final features
        - Aggregated player snapshots
        - Removed leakage-prone variables
        """)
    
    with tab3:

        st.markdown("""
        Models evaluated:

        - Logistic Regression
        - Random Forest
        - XGBoost
        - LightGBM

        Logistic Regression achieved the highest test ROC-AUC (0.833).
        """)

    with tab4:

        st.markdown("""
        SHAP analysis identified the strongest churn drivers:

        - Activity span
        - Observation count
        - Zone diversity
        - Guild participation
        """)
    st.divider()

    st.subheader("🔍 Key Findings")

    st.success(
        "Players active for longer periods are substantially less likely to churn."
    )

    st.success(
        "Guild participation is strongly associated with player retention."
    )

    st.success(
        "Exploring more zones correlates with lower churn risk."
    )

    st.success(
        "Low progression and limited activity are major warning signs."
    )

    st.divider()

    st.subheader("💻 Technology Stack")

    st.markdown("""
    - Python
    - Pandas
    - NumPy
    - Scikit-learn
    - XGBoost
    - LightGBM
    - SHAP
    - Plotly
    - Streamlit
    """)
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="WoW Churn Analytics",
        page_icon="🎮",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
    <style>

    /* Main layout */
    .block-container{
        padding-top:2rem;
        padding-bottom:2rem;
    }

    /* Metric cards */
    [data-testid="stMetric"]{
        background-color:#ffffff;
        border:1px solid #e5e7eb;
        padding:15px;
        border-radius:12px;
        box-shadow:0px 2px 8px rgba(0,0,0,0.05);
    }

    /* Headers */
    h1{
        color:#111827;
    }

    h2,h3{
        color:#1f2937;
    }

    /* Risk cards */
    .risk-card{
        padding:20px;
        border-radius:12px;
        text-align:center;
        font-size:18px;
        font-weight:bold;
    }

    /* Dashboard cards */
    .insight-card{
        background:#f8fafc;
        border-left:5px solid #2563eb;
        padding:15px;
        border-radius:10px;
        margin-bottom:10px;
    }

    </style>
    """, unsafe_allow_html=True)

    # ====================================
    # HERO SECTION
    # ====================================

    st.markdown("""
    <div style="
        padding:25px;
        border-radius:15px;
        background:linear-gradient(135deg,#0f172a,#1e293b);
        color:white;
        margin-bottom:20px;
    ">
        <h1 style="color:white;">
            🎮 World of Warcraft Churn Analytics
        </h1>
        <p style="font-size:18px;">
            Predict player retention using machine learning,
            behavioral analytics, and explainable AI.
        </p>
    </div>
    """, unsafe_allow_html=True)

    hero1, hero2, hero3 = st.columns(3)

    with hero1:
        st.metric(
            "Dataset Size",
            "1.38M"
        )

    with hero2:
        st.metric(
            "Characters",
            "1,195"
        )

    with hero3:
        st.metric(
            "ROC-AUC",
            "0.833"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    try:
        model, scaler = load_model()
        ref = load_reference_data()
        thresholds = derive_thresholds(ref)
    except Exception as e:
        st.error(f"Could not load model artifacts: {e}")
        st.stop()

    page = option_menu(
        menu_title=None,
        options=[
            "Dashboard",
            "Prediction",
            "Batch Analytics",
            "Performance",
            "About"
        ],
        icons=[
            "house",
            "magic",
            "bar-chart",
            "graph-up",
            "info-circle"
        ],
        orientation="horizontal",
        default_index=0
    )

    st.divider()

    if page == "Dashboard":
        page_dashboard()

    elif page == "Prediction":
        page_single(
            model,
            scaler,
            ref,
            thresholds
        )

    elif page == "Batch Analytics":
        page_batch(
            model,
            scaler,
            ref,
            thresholds
        )

    elif page == "Performance":
        page_performance()

    elif page == "About":
        page_about()


if __name__ == "__main__":
    main()
