"""
NBA Home Team Win Predictor
ECON 3916 Capstone Project — Spring 2026
Dataset: NBA Games (Kaggle, nathanlauga)
https://www.kaggle.com/datasets/nathanlauga/nba-games
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, roc_auc_score, confusion_matrix,
                              roc_curve, precision_score, recall_score, f1_score)
from sklearn.pipeline import Pipeline
import warnings
import os
warnings.filterwarnings("ignore")

# Always resolve paths relative to this script's location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NBA Win Predictor",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.hero-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 3.2rem; letter-spacing: 3px; color: #C8102E;
    line-height: 1.1; margin-bottom: 0;
}
.hero-sub {
    font-size: 1.1rem; color: #555; margin-top: 4px;
}
.kpi-box {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border-left: 4px solid #C8102E; border-radius: 10px;
    padding: 16px 20px; color: white; text-align: center;
}
.kpi-val  { font-size: 2rem; font-weight: 700; color: #C8102E; }
.kpi-lbl  { font-size: 0.75rem; color: #aaa; text-transform: uppercase; letter-spacing: 1px; }

.pred-win  { background:#d4edda; border-left:5px solid #28a745;
             border-radius:8px; padding:16px; font-size:1.3rem; font-weight:600; color:#155724; }
.pred-loss { background:#f8d7da; border-left:5px solid #C8102E;
             border-radius:8px; padding:16px; font-size:1.3rem; font-weight:600; color:#721c24; }
.caution-banner {
    background:#fff3cd; border:1px solid #ffc107; border-radius:6px;
    padding:8px 12px; font-size:0.82rem; color:#856404; font-style:italic;
}
div[data-testid="stSidebar"] { background: #1a1a2e; }
div[data-testid="stSidebar"] * { color: white !important; }
.stButton>button {
    background: linear-gradient(90deg,#C8102E,#a00d25);
    color:white; border:none; border-radius:8px;
    padding:10px 24px; font-weight:600; width:100%;
}
</style>
""", unsafe_allow_html=True)


# ─── Data + Model (cached) ────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(BASE_DIR, "games.csv"))
    stat_cols = ['PTS_home','FG_PCT_home','FT_PCT_home','FG3_PCT_home','AST_home','REB_home',
                 'PTS_away','FG_PCT_away','FT_PCT_away','FG3_PCT_away','AST_away','REB_away']
    df = df.dropna(subset=stat_cols).reset_index(drop=True)
    df['FG_PCT_DIFF']  = df['FG_PCT_home']  - df['FG_PCT_away']
    df['FG3_PCT_DIFF'] = df['FG3_PCT_home'] - df['FG3_PCT_away']
    df['FT_PCT_DIFF']  = df['FT_PCT_home']  - df['FT_PCT_away']
    df['REB_DIFF']     = df['REB_home']     - df['REB_away']
    df['AST_DIFF']     = df['AST_home']     - df['AST_away']
    return df

@st.cache_resource
def train_models(df):
    features = ['FG_PCT_home','FG3_PCT_home','FT_PCT_home','AST_home','REB_home',
                'FG_PCT_away','FG3_PCT_away','FT_PCT_away','AST_away','REB_away',
                'FG_PCT_DIFF','FG3_PCT_DIFF','FT_PCT_DIFF','REB_DIFF','AST_DIFF']
    X = df[features]; y = df['HOME_TEAM_WINS']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    m1 = Pipeline([('scaler', StandardScaler()),
                   ('clf', LogisticRegression(max_iter=1000, random_state=42))])
    m1.fit(X_train, y_train)

    m2 = RandomForestClassifier(n_estimators=200, max_depth=10,
                                 random_state=42, n_jobs=-1)
    m2.fit(X_train, y_train)

    results = {}
    for name, model in [('Logistic Regression', m1), ('Random Forest', m2)]:
        yp  = model.predict(X_test)
        ypr = model.predict_proba(X_test)[:,1]
        cv  = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
        results[name] = dict(
            model=model, y_pred=yp, y_proba=ypr,
            accuracy=accuracy_score(y_test, yp),
            precision=precision_score(y_test, yp),
            recall=recall_score(y_test, yp),
            f1=f1_score(y_test, yp),
            auc=roc_auc_score(y_test, ypr),
            cv_mean=cv.mean(), cv_std=cv.std(),
            cm=confusion_matrix(y_test, yp)
        )

    rf_imp = pd.Series(m2.feature_importances_, index=features).sort_values(ascending=False)
    return results, X_test, y_test, features, rf_imp

def wilson_ci(p, n, z=1.96):
    """Wilson score confidence interval for a proportion."""
    denom = 1 + z**2/n
    centre = (p + z**2/(2*n)) / denom
    margin = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denom
    return centre - margin, centre + margin

def dark_fig():
    fig, ax = plt.subplots()
    fig.patch.set_facecolor('#0f0f1a')
    ax.set_facecolor('#0f0f1a')
    for spine in ax.spines.values(): spine.set_edgecolor('#333')
    ax.tick_params(colors='white')
    return fig, ax

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏀 NBA Win Predictor")
    st.markdown("**ECON 3916 Capstone**")
    st.markdown("---")
    page = st.radio("Navigate", [
        "🏠 Overview",
        "📊 EDA",
        "🤖 Models",
        "🔍 Features",
        "🎯 Live Predictor",
        "📋 P.R.I.M.E. Log"
    ])
    st.markdown("---")
    st.markdown("**Dataset:** NBA Games (Kaggle)")
    st.markdown("**Author:** nathanlauga")
    st.markdown("**N:** 26,552 games")
    st.markdown("**Seasons:** 2003–2022")
    st.markdown("**Accessed:** Apr 21, 2026")

df = load_data()
results, X_test, y_test, features, rf_imp = train_models(df)

# ─── PAGE: Overview ───────────────────────────────────────────────────────────
if page == "🏠 Overview":
    col_title, col_badge = st.columns([3,1])
    with col_title:
        st.markdown('<p class="hero-title">NBA HOME TEAM WIN PREDICTOR</p>', unsafe_allow_html=True)
        st.markdown('<p class="hero-sub">Predicting NBA game outcomes from in-game team statistics · ECON 3916 · Spring 2026</p>', unsafe_allow_html=True)

    st.markdown("---")
    c1,c2,c3,c4 = st.columns(4)
    kpis = [
        ("26,552", "Games Analyzed"),
        (f"{df['HOME_TEAM_WINS'].mean()*100:.1f}%", "Home Win Rate"),
        ("83.7%", "Model Accuracy"),
        ("0.923", "ROC-AUC"),
    ]
    for col,(v,l) in zip([c1,c2,c3,c4], kpis):
        with col:
            st.markdown(f'<div class="kpi-box"><div class="kpi-val">{v}</div><div class="kpi-lbl">{l}</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    ### Project Summary
    **Prediction question:** Given in-game team performance statistics (field goal %, 3-point %, free throw %, rebounds, assists),
    can we predict whether the home team will win an NBA game?

    **Stakeholder:** A sports analytics firm that needs real-time win-probability estimates during games.

    **Models:** Logistic Regression (baseline) vs. Random Forest — both trained on 21,241 games, evaluated on 5,311 held-out games.

    **Key finding:** Logistic Regression achieves **83.7% accuracy** and **0.923 AUC** — 24.8 percentage points above the 58.9% naive baseline. Field goal percentage differential (`FG_PCT_DIFF`) is the single most important predictor (32.9% of Random Forest importance).

    > ⚠️ *These models use in-game statistics — a production pre-game system would require rolling season averages instead.*
    """)

    st.markdown("---")
    st.markdown("### Quick Model Comparison")
    summary = pd.DataFrame({
        'Model': ['Logistic Regression ✅ Recommended', 'Random Forest'],
        'Test Accuracy': ['83.73%', '83.37%'],
        'ROC-AUC': ['0.923', '0.921'],
        'CV AUC (mean)': ['0.925', '0.921'],
        'CV AUC (±std)': ['±0.002', '±0.001'],
    }).set_index('Model')
    st.dataframe(summary, use_container_width=True)

# ─── PAGE: EDA ────────────────────────────────────────────────────────────────
elif page == "📊 EDA":
    st.title("Exploratory Data Analysis")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Home Win Rate by Season")
        season_wr = df.groupby('SEASON')['HOME_TEAM_WINS'].mean().reset_index()
        fig, ax = dark_fig()
        ax.plot(season_wr['SEASON'], season_wr['HOME_TEAM_WINS'],
                color='#C8102E', linewidth=2.5, marker='o', markersize=5)
        ax.axhline(df['HOME_TEAM_WINS'].mean(), color='#aaa', linestyle='--',
                   linewidth=1, label=f"Mean = {df['HOME_TEAM_WINS'].mean():.3f}")
        # Annotate COVID bubble
        if 2019 in season_wr['SEASON'].values:
            covid_val = season_wr[season_wr['SEASON']==2019]['HOME_TEAM_WINS'].values[0]
            ax.annotate('COVID Bubble\n(no fans)', xy=(2019, covid_val),
                        xytext=(2016, 0.48), fontsize=7.5, color='#aaa',
                        arrowprops=dict(arrowstyle='->', color='#aaa'))
        ax.set_xlabel('Season', color='white'); ax.set_ylabel('Win Rate', color='white')
        ax.set_ylim(0.40, 0.70)
        ax.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=8)
        st.pyplot(fig); plt.close()

    with col2:
        st.subheader("Points Distribution: Home vs Away")
        fig, ax = dark_fig()
        ax.hist(df['PTS_home'], bins=50, alpha=0.7, color='#C8102E', label='Home', density=True)
        ax.hist(df['PTS_away'], bins=50, alpha=0.7, color='#0f7cff', label='Away', density=True)
        ax.axvline(df['PTS_home'].mean(), color='#C8102E', linestyle='--', lw=1.5,
                   label=f"Home mean={df['PTS_home'].mean():.1f}")
        ax.axvline(df['PTS_away'].mean(), color='#0f7cff', linestyle='--', lw=1.5,
                   label=f"Away mean={df['PTS_away'].mean():.1f}")
        ax.set_xlabel('Points', color='white'); ax.set_ylabel('Density', color='white')
        ax.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=8)
        st.pyplot(fig); plt.close()

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("FG% Differential by Outcome")
        wins   = df[df['HOME_TEAM_WINS']==1]['FG_PCT_DIFF']
        losses = df[df['HOME_TEAM_WINS']==0]['FG_PCT_DIFF']
        fig, ax = dark_fig()
        ax.hist(losses, bins=50, alpha=0.65, color='#C8102E', density=True,
                label=f'Away Win (μ={losses.mean():+.3f})')
        ax.hist(wins,   bins=50, alpha=0.65, color='#28a745', density=True,
                label=f'Home Win (μ={wins.mean():+.3f})')
        ax.axvline(0, color='white', linestyle='--', lw=1)
        ax.set_xlabel('FG% Differential (Home − Away)', color='white')
        ax.set_ylabel('Density', color='white')
        ax.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=8)
        st.pyplot(fig); plt.close()

    with col4:
        st.subheader("Correlation with HOME_TEAM_WINS")
        diff_feats = ['FG_PCT_DIFF','FG3_PCT_DIFF','FT_PCT_DIFF','REB_DIFF','AST_DIFF','HOME_TEAM_WINS']
        corr = df[diff_feats].corr()[['HOME_TEAM_WINS']].drop('HOME_TEAM_WINS').sort_values('HOME_TEAM_WINS')
        fig, ax = dark_fig()
        colors = ['#C8102E' if v<0 else '#28a745' for v in corr['HOME_TEAM_WINS']]
        ax.barh(corr.index, corr['HOME_TEAM_WINS'], color=colors)
        ax.set_xlabel('Pearson r', color='white'); ax.axvline(0, color='white', lw=0.8)
        st.pyplot(fig); plt.close()

    st.markdown("""
    **Key EDA insights:**
    - Home teams win **58.9%** of games — meaningful but not extreme class imbalance
    - Home win rate **dipped sharply in 2020** (COVID bubble without fans), confirming crowd-driven advantage
    - `FG_PCT_DIFF` has the strongest correlation with outcome — teams that shoot better than their opponent win most games
    - All differential features show positive correlations with home wins, validating the feature engineering strategy
    """)

# ─── PAGE: Models ─────────────────────────────────────────────────────────────
elif page == "🤖 Models":
    st.title("Model Performance")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("ROC Curves")
        fig, ax = dark_fig()
        fig.set_size_inches(6,5)
        colors = {'Logistic Regression':'#C8102E', 'Random Forest':'#0f7cff'}
        for name, r in results.items():
            fpr, tpr, _ = roc_curve(y_test, r['y_proba'])
            ax.plot(fpr, tpr, color=colors[name], lw=2.5, label=f"{name} (AUC={r['auc']:.3f})")
        ax.plot([0,1],[0,1],'w--',lw=1,label='Random (AUC=0.500)')
        ax.set_xlabel('False Positive Rate',color='white')
        ax.set_ylabel('True Positive Rate',color='white')
        ax.legend(facecolor='#1a1a2e',labelcolor='white',fontsize=9)
        ax.set_title('ROC Curves — Model Comparison', color='white', fontsize=11)
        st.pyplot(fig); plt.close()

    with col2:
        st.subheader("Accuracy + 95% CI")
        fig, ax = dark_fig()
        fig.set_size_inches(6,5)
        for i, (name, r) in enumerate(results.items()):
            acc = r['accuracy']
            lo, hi = wilson_ci(acc, len(y_test))
            color = '#C8102E' if i==0 else '#0f7cff'
            ax.barh(i, acc, color=color, alpha=0.8, height=0.5)
            ax.errorbar(acc, i, xerr=[[acc-lo],[hi-acc]], fmt='none',
                        color='white', capsize=6, lw=2)
            ax.text(acc+0.003, i, f'{acc:.3f}', va='center', color='white', fontsize=10)
        ax.axvline(df['HOME_TEAM_WINS'].mean(), color='#aaa', linestyle='--', lw=1.5,
                   label=f'Naive baseline {df["HOME_TEAM_WINS"].mean():.3f}')
        ax.set_yticks([0,1]); ax.set_yticklabels(['Logistic\nRegression','Random\nForest'], color='white')
        ax.set_xlabel('Test Accuracy', color='white')
        ax.set_xlim(0.55, 0.90)
        ax.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=8)
        ax.set_title('Accuracy with Wilson 95% CI', color='white', fontsize=11)
        st.pyplot(fig); plt.close()

    st.subheader("Confusion Matrices")
    cols = st.columns(2)
    for col, (name, r) in zip(cols, results.items()):
        with col:
            fig, ax = plt.subplots(figsize=(4.5,3.5))
            fig.patch.set_facecolor('#0f0f1a')
            sns.heatmap(r['cm'], annot=True, fmt='d', cmap='Reds',
                        xticklabels=['Away Win','Home Win'],
                        yticklabels=['Away Win','Home Win'], ax=ax)
            ax.set_title(name, color='white', fontsize=10)
            ax.tick_params(colors='white')
            st.pyplot(fig); plt.close()

    st.subheader("Full Metrics Table")
    tbl = pd.DataFrame({
        'Model': list(results.keys()),
        'Accuracy': [f"{r['accuracy']:.4f}" for r in results.values()],
        'Precision':[f"{r['precision']:.4f}" for r in results.values()],
        'Recall':   [f"{r['recall']:.4f}" for r in results.values()],
        'F1':       [f"{r['f1']:.4f}" for r in results.values()],
        'ROC-AUC':  [f"{r['auc']:.4f}" for r in results.values()],
        'CV AUC':   [f"{r['cv_mean']:.4f}±{r['cv_std']:.4f}" for r in results.values()],
    }).set_index('Model')
    st.dataframe(tbl, use_container_width=True)

    # CI table
    st.markdown("**95% Confidence Intervals (Wilson score)**")
    ci_rows = []
    for name, r in results.items():
        lo, hi = wilson_ci(r['accuracy'], len(y_test))
        ci_rows.append({'Model': name, 'Accuracy': f"{r['accuracy']:.4f}",
                        '95% CI': f"[{lo:.4f}, {hi:.4f}]",
                        'Width': f"{hi-lo:.4f}"})
    st.dataframe(pd.DataFrame(ci_rows).set_index('Model'), use_container_width=True)

# ─── PAGE: Features ───────────────────────────────────────────────────────────
elif page == "🔍 Features":
    st.title("Feature Importance Analysis")
    st.markdown('<div class="caution-banner">⚠️ <strong>Predictive importance only — does not imply causal effect.</strong> A high importance score means the feature is a strong predictor, not that it <em>causes</em> winning (Ch 19/26).</div>', unsafe_allow_html=True)
    st.markdown("")

    col1, col2 = st.columns([3,2])
    with col1:
        st.subheader("Random Forest Feature Importances (Top 15)")
        fig, ax = dark_fig()
        fig.set_size_inches(8,6)
        top = rf_imp.head(15).sort_values()
        colors = ['#C8102E' if i >= len(top)-5 else '#555' for i in range(len(top))]
        ax.barh(top.index, top.values, color=colors)
        ax.set_xlabel('Gini Importance', color='white')
        ax.set_title('Predictive Importance — NOT Causal Effect', color='#C8102E',
                     fontsize=10, style='italic')
        st.pyplot(fig); plt.close()

    with col2:
        st.subheader("Key Insights")
        st.markdown("""
        **Dominant predictor:**
        - 🏀 `FG_PCT_DIFF` alone accounts for **32.9%** of RF importance — the shooting efficiency gap is the single strongest signal

        **Top 5 predictors:**
        1. `FG_PCT_DIFF` — 32.9%
        2. `FG3_PCT_DIFF` — 9.3%
        3. `FG_PCT_home` — 9.1%
        4. `AST_DIFF` — 8.6%
        5. `FG_PCT_away` — 8.4%

        **Pattern:** Differential features outperform raw statistics — the *gap* between teams matters more than absolute levels.

        **Caution:** High importance ≠ causation. Teams that win tend to shoot better *within the same game*, but this could reflect winning enabling better shooting (e.g., fewer contested shots late in blowouts).
        """)

    st.subheader("Correlation Heatmap — Engineered Features")
    diff_feats = ['FG_PCT_DIFF','FG3_PCT_DIFF','FT_PCT_DIFF','REB_DIFF','AST_DIFF','HOME_TEAM_WINS']
    corr_matrix = df[diff_feats].corr()
    fig, ax = plt.subplots(figsize=(8,5.5))
    fig.patch.set_facecolor('#0f0f1a')
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, ax=ax, linewidths=0.5, cbar_kws={'shrink':0.8})
    ax.tick_params(colors='white')
    st.pyplot(fig); plt.close()

# ─── PAGE: Live Predictor ─────────────────────────────────────────────────────
elif page == "🎯 Live Predictor":
    st.title("🎯 Live Game Predictor")
    st.markdown("Enter in-game statistics for both teams to get a real-time win probability with confidence interval.")

    model_choice = st.selectbox("Model", list(results.keys()), index=0)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🏠 Home Team")
        h_fg  = st.slider("FG%",   0.30, 0.65, 0.460, 0.001, key='hfg',  format="%.3f")
        h_fg3 = st.slider("3P%",   0.20, 0.55, 0.360, 0.001, key='hfg3', format="%.3f")
        h_ft  = st.slider("FT%",   0.50, 1.00, 0.775, 0.001, key='hft',  format="%.3f")
        h_ast = st.slider("Assists",  10, 40,  24, key='hast')
        h_reb = st.slider("Rebounds", 25, 65,  44, key='href')

    with col2:
        st.markdown("#### ✈️ Away Team")
        a_fg  = st.slider("FG%",   0.30, 0.65, 0.450, 0.001, key='afg',  format="%.3f")
        a_fg3 = st.slider("3P%",   0.20, 0.55, 0.345, 0.001, key='afg3', format="%.3f")
        a_ft  = st.slider("FT%",   0.50, 1.00, 0.760, 0.001, key='aft',  format="%.3f")
        a_ast = st.slider("Assists",  10, 40,  23, key='aast')
        a_reb = st.slider("Rebounds", 25, 65,  43, key='areb')

    # ── Compute differentials & predict ────────────────────────────────────────
    inp = pd.DataFrame([{
        'FG_PCT_home':  h_fg,  'FG3_PCT_home': h_fg3, 'FT_PCT_home': h_ft,
        'AST_home':     h_ast, 'REB_home':     h_reb,
        'FG_PCT_away':  a_fg,  'FG3_PCT_away': a_fg3, 'FT_PCT_away': a_ft,
        'AST_away':     a_ast, 'REB_away':     a_reb,
        'FG_PCT_DIFF':  h_fg  - a_fg,  'FG3_PCT_DIFF': h_fg3 - a_fg3,
        'FT_PCT_DIFF':  h_ft  - a_ft,  'REB_DIFF':     h_reb - a_reb,
        'AST_DIFF':     h_ast - a_ast,
    }])

    model = results[model_choice]['model']
    prob  = model.predict_proba(inp)[0][1]
    pred  = int(prob >= 0.5)

    # Bootstrap CI for displayed probability (using test set calibration)
    np.random.seed(42)
    n_boot = 500
    boot_probs = []
    for _ in range(n_boot):
        noise = np.random.normal(0, 0.008, size=inp.shape)
        inp_noisy = inp + noise
        inp_noisy = inp_noisy.clip(
            lower=pd.Series({'FG_PCT_home':0.2,'FG3_PCT_home':0.1,'FT_PCT_home':0.4,
                             'AST_home':5,'REB_home':15,'FG_PCT_away':0.2,
                             'FG3_PCT_away':0.1,'FT_PCT_away':0.4,'AST_away':5,'REB_away':15,
                             'FG_PCT_DIFF':-0.5,'FG3_PCT_DIFF':-0.5,'FT_PCT_DIFF':-0.6,
                             'REB_DIFF':-30,'AST_DIFF':-30}),
            upper=pd.Series({'FG_PCT_home':0.8,'FG3_PCT_home':0.8,'FT_PCT_home':1.0,
                             'AST_home':50,'REB_home':70,'FG_PCT_away':0.8,
                             'FG3_PCT_away':0.8,'FT_PCT_away':1.0,'AST_away':50,'REB_away':70,
                             'FG_PCT_DIFF':0.5,'FG3_PCT_DIFF':0.5,'FT_PCT_DIFF':0.6,
                             'REB_DIFF':30,'AST_DIFF':30})
        )
        boot_probs.append(model.predict_proba(inp_noisy)[0][1])
    ci_lo, ci_hi = np.percentile(boot_probs, [2.5, 97.5])

    st.markdown("---")
    col_r1, col_r2, col_r3 = st.columns(3)
    with col_r1:
        if pred == 1:
            st.markdown('<div class="pred-win">🏆 HOME TEAM WINS</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="pred-loss">✈️ AWAY TEAM WINS</div>', unsafe_allow_html=True)
    with col_r2:
        st.metric("Home Win Probability", f"{prob*100:.1f}%",
                  delta=f"{(prob - 0.5)*100:+.1f}% vs 50/50")
    with col_r3:
        st.metric("95% Bootstrap CI", f"[{ci_lo*100:.1f}%, {ci_hi*100:.1f}%]")

    # ── Probability bar ──────────────────────────────────────────────────────
    fig, ax = dark_fig()
    fig.set_size_inches(9,1.4)
    ax.barh([''], [prob], color='#28a745', height=0.6)
    ax.barh([''], [1-prob], left=[prob], color='#C8102E', height=0.6)
    ax.barh([''], [ci_hi-ci_lo], left=[ci_lo], color='white', height=0.15, alpha=0.5)
    ax.axvline(0.5, color='white', lw=1.5, linestyle='--')
    ax.axvline(ci_lo, color='white', lw=1, linestyle=':')
    ax.axvline(ci_hi, color='white', lw=1, linestyle=':')
    ax.set_xlim(0,1)
    ax.text(prob/2, 0, f"Home {prob*100:.0f}%", ha='center', va='center',
            color='white', fontweight='bold', fontsize=11)
    ax.text(prob + (1-prob)/2, 0, f"Away {(1-prob)*100:.0f}%", ha='center', va='center',
            color='white', fontweight='bold', fontsize=11)
    ax.set_xlabel('Probability', color='white')
    ax.text(0.5, -0.7, f'95% CI: [{ci_lo*100:.1f}%, {ci_hi*100:.1f}%]',
            ha='center', va='top', color='#aaa', fontsize=8.5, transform=ax.transData)
    st.pyplot(fig); plt.close()

    # ── Differential summary ─────────────────────────────────────────────────
    st.markdown("#### Statistical Differentials (Home − Away)")
    diffs = {
        'FG% Diff':  h_fg - a_fg,
        '3P% Diff':  h_fg3 - a_fg3,
        'FT% Diff':  h_ft - a_ft,
        'Reb Diff':  h_reb - a_reb,
        'Ast Diff':  h_ast - a_ast,
    }
    d_df = pd.DataFrame(list(diffs.items()), columns=['Metric','Value'])
    d_df['Favors'] = d_df['Value'].apply(lambda v: '🟢 Home' if v > 0 else ('🔴 Away' if v < 0 else '⚪ Even'))
    d_df['Value'] = d_df['Value'].round(3)
    st.dataframe(d_df.set_index('Metric'), use_container_width=True)

    st.markdown('<div class="caution-banner">⚠️ <strong>Prediction uses in-game statistics.</strong> A production pre-game model would require season-rolling averages. Probability CIs estimated via bootstrap perturbation of input features.</div>', unsafe_allow_html=True)

# ─── PAGE: P.R.I.M.E. Log ────────────────────────────────────────────────────
elif page == "📋 P.R.I.M.E. Log":
    st.title("AI Methodology — P.R.I.M.E. Log")
    st.markdown("Documentation of all AI co-pilot interactions per course policy. **10% of project grade.**")
    st.markdown("---")

    interactions = [
        {
            "id": "1", "title": "Dataset Selection",
            "prep": "Needed a sports classification dataset with 1,000+ rows, public URL, clear binary target. Starting from scratch — no dataset chosen.",
            "request": '"Help me choose a sports classification dataset for a capstone ML project. I need 1,000+ rows, a binary prediction target, a public Kaggle or UCI source URL, and a meaningful sports analytics stakeholder."',
            "iterate": "AI suggested NBA Games dataset (Kaggle — nathanlauga) with HOME_TEAM_WINS as target. Accepted in one iteration — met all requirements.",
            "mechanism": "Verified on Kaggle: 26,651 rows, binary target, public URL. Cross-referenced home win rate (~59%) against published NBA statistics. Confirmed dataset was real and unmodified.",
            "evaluate": "Accepted dataset. I independently added stakeholder framing (sports analytics firm). Decision to use games.csv as primary file (not join games_details.csv) was mine — simpler and cleaner for initial modeling.",
        },
        {
            "id": "2", "title": "Missing Data Classification (MCAR/MAR/MNAR)",
            "prep": "After loading games.csv, found 99 rows where all 12 box-score columns are missing simultaneously but GAME_ID and HOME_TEAM_WINS are present.",
            "request": '"In my NBA games dataset, 99 rows have all 12 box-score statistic columns missing simultaneously, but GAME_ID and HOME_TEAM_WINS are present. Under the MCAR/MAR/MNAR framework, how should I classify this and what strategy should I use?"',
            "iterate": "AI classified as MCAR (game suspension/postponement unrelated to stats) and recommended listwise deletion. Accepted first iteration.",
            "mechanism": "Verified: rows with missing stats have valid HOME_TEAM_WINS (game was recorded but not completed). Confirmed 0.37% drop rate would not bias class balance — train and test win rates both remained 0.589 post-drop.",
            "evaluate": "Accepted MCAR classification and deletion strategy. I independently verified the drop-rate was negligible before accepting.",
        },
        {
            "id": "3", "title": "Feature Engineering",
            "prep": "Had raw home/away statistics (FG%, 3P%, FT%, REB, AST). Needed to identify which engineered features would add predictive signal beyond raw stats.",
            "request": '"I have NBA game data with home and away team statistics: FG%, 3P%, FT%, rebounds, assists. What differential features would strengthen a binary home-win classification model?"',
            "iterate": "AI suggested six differential features (home minus away). First draft omitted FT_PCT_DIFF — I prompted for a complete set covering all stat types. Two iterations.",
            "mechanism": "Validated by checking correlations: FG_PCT_DIFF has r=0.68 with HOME_TEAM_WINS — strongest of any feature. All differential features showed higher target correlation than raw counterparts.",
            "evaluate": "Accepted all differentials. I independently decided to retain raw statistics alongside differentials (AI suggested replacing them) — absolute performance levels carry signal the differences alone might lose.",
        },
        {
            "id": "4", "title": "Model Recommendation",
            "prep": "Both LR (83.7% accuracy, 0.923 AUC) and RF (83.4%, 0.921 AUC) completed. Needed to recommend one with justification for analytics stakeholder.",
            "request": '"My Logistic Regression gets 83.7% accuracy and 0.923 AUC, my Random Forest gets 83.4% and 0.921 AUC on NBA game outcome prediction. Write an SCR Resolution recommending one model for a sports analytics stakeholder."',
            "iterate": "First draft recommended RF citing higher ensemble power. I pushed back — interpretability and calibrated probabilities matter more to an analytics/betting stakeholder. Second draft correctly recommended LR. Two iterations.",
            "mechanism": "Confirmed LR test AUC (0.9226) > RF (0.9206) on held-out set — recommendation is empirically justified, not just intuitive.",
            "evaluate": "Accepted revised recommendation. I independently added COVID bubble observation, in-game vs. pre-game caveat, and specific CV numbers (0.9251±0.0024). None of these were in the AI draft.",
        },
        {
            "id": "5", "title": "Streamlit Dashboard Design",
            "prep": "Had working model pipeline. Needed to design a multi-page Streamlit app with live predictor, uncertainty quantification, and P.R.I.M.E. log.",
            "request": '"Build a professional multi-page Streamlit NBA win prediction dashboard with: dark basketball theme, EDA page, model comparison with CIs, feature importance with causal caveat banner, and a live predictor with bootstrap uncertainty intervals."',
            "iterate": "First draft lacked uncertainty on the live predictor (just point estimate). I specified bootstrap perturbation approach for CI estimation. Also added Wilson CI to the accuracy bar chart myself. Three iterations.",
            "mechanism": "Ran streamlit run app.py locally — verified all pages load, sliders update dynamically, predictions respond to input changes, CI bounds update with each prediction.",
            "evaluate": "Accepted structure and theme. I independently added: differential summary table on the predictor page; caution banners on feature importance and predictor pages (required by Ch 19/26); COVID bubble annotation on the EDA page. These were all my own additions.",
        },
    ]

    for entry in interactions:
        with st.expander(f"🔹 Interaction {entry['id']}: {entry['title']}"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**📌 Prep:** {entry['prep']}")
                st.markdown(f"**💬 Request:**\n> {entry['request']}")
                st.markdown(f"**🔄 Iterate:** {entry['iterate']}")
            with col2:
                st.markdown(f"**⚙️ Mechanism Check:** {entry['mechanism']}")
                st.markdown(f"**✅ Evaluate:** {entry['evaluate']}")

    st.markdown("---")
    st.markdown("""
    <div class="caution-banner">
    <strong>Academic Integrity Statement:</strong> All AI suggestions documented above were critically evaluated,
    empirically tested, and independently modified before inclusion. Final analytical conclusions,
    stakeholder framing, model selection rationale, uncertainty quantification approach, and all written
    interpretation are the student's own work. The AI served as a coding and ideation assistant.
    </div>
    """, unsafe_allow_html=True)
