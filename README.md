# NBA Home Team Win Predictor
### ECON 3916 — Applied Machine Learning | Capstone Project | Spring 2026

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://nba-win-predictor.streamlit.app)

> **Prediction question:** Given in-game team performance statistics (field goal %, 3-point %, free throw %, rebounds, assists), can we predict whether the home team will win an NBA game?

---

## Project Overview

| Item | Detail |
|---|---|
| **Dataset** | NBA Games — Kaggle (nathanlauga) |
| **Source URL** | https://www.kaggle.com/datasets/nathanlauga/nba-games |
| **Access date** | April 21, 2026 |
| **N** | 26,552 games (after cleaning) |
| **Seasons** | 2003–2022 |
| **Target** | `HOME_TEAM_WINS` (binary) |
| **Best model** | Logistic Regression — 83.7% accuracy, 0.923 AUC |

---

## Repository Structure

```
nba-win-predictor/
├── app.py                              # Streamlit dashboard
├── requirements.txt                    # Python dependencies (pinned)
├── README.md                           # This file
├── 3916-final-project-completed.ipynb  # Full analysis notebook
└── data/
    ├── games.csv           # Primary modeling file (one row per game)
    ├── games_details.csv   # Player-level box scores
    ├── teams.csv           # Team metadata
    ├── players.csv         # Player metadata
    └── ranking.csv         # Season standings
```

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/nba-win-predictor.git
cd nba-win-predictor
```

### 2. Set up the Python environment

**Using conda (recommended):**
```bash
conda create -n nba-predictor python=3.10 -y
conda activate nba-predictor
pip install -r requirements.txt
```

**Using venv:**
```bash
python -m venv venv
source venv/bin/activate          # Mac/Linux
# venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

### 3. Data acquisition

The `data/` folder must contain the five CSV files from the Kaggle dataset. Two options:

**Option A — Download from Kaggle (requires free account):**
```bash
pip install kaggle
kaggle datasets download -d nathanlauga/nba-games -p data/ --unzip
```

**Option B — Manual download:**
1. Go to https://www.kaggle.com/datasets/nathanlauga/nba-games
2. Click "Download" (free account required)
3. Extract all CSV files into the `data/` folder

The required files are:
- `data/games.csv` — **required for modeling**
- `data/games_details.csv`
- `data/teams.csv`
- `data/players.csv`
- `data/ranking.csv`

### 4. Run the Jupyter notebook

```bash
jupyter notebook 3916-final-project-completed.ipynb
```

Run all cells top to bottom. The notebook will:
- Load `data/games.csv`
- Engineer differential features
- Train Logistic Regression and Random Forest models
- Produce all EDA visualizations and model evaluation outputs

Expected runtime: ~2–3 minutes (Random Forest with 200 trees)

### 5. Launch the Streamlit app locally

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`. Verify all 6 pages load correctly before deploying.

---

## Reproducing Results

All random operations use `random_state=42`. With the same data and environment, you will get:

| Metric | Value |
|---|---|
| Logistic Regression Accuracy | 0.8373 |
| Logistic Regression ROC-AUC | 0.9226 |
| Logistic Regression CV AUC | 0.9251 ± 0.0024 |
| Random Forest Accuracy | 0.8337 |
| Random Forest ROC-AUC | 0.9206 |
| Random Forest CV AUC | 0.9209 ± 0.0013 |
| Home win rate (full dataset) | 0.589 |
| Training set size | 21,241 games |
| Test set size | 5,311 games |

---

## Streamlit Deployment (Streamlit Community Cloud)

1. Push all files to a **public GitHub repository** (or private with Streamlit access)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub → click **"New app"**
4. Select your repo, branch (`main`), and set main file to `app.py`
5. Click **"Deploy"**
6. Your app will be live at `https://YOUR-APP-NAME.streamlit.app`

> **Note:** Streamlit Community Cloud must be able to access your `data/` folder. Include `data/games.csv` in your repo (file size ~5 MB — within GitHub's limits).

---

## Key Findings

1. **Logistic Regression is the recommended model** — higher test AUC (0.923 vs 0.921), calibrated probability outputs, and interpretable coefficients for the analytics stakeholder
2. **FG_PCT_DIFF dominates** — field goal percentage differential accounts for 32.9% of Random Forest feature importance; the shooting efficiency gap is the single strongest predictor
3. **Home court advantage is crowd-driven** — the 2020 COVID bubble season (no fans) shows a clear dip in home win rate, confirming crowd presence as a structural driver
4. **In-game limitation** — all features are recorded during the game being predicted; a production pre-game system requires season-rolling statistics instead

---

## Limitations

- Features are in-game statistics, not pre-game predictors
- No temporal holdout (random split may allow future games to inform past)
- Missing contextual features: player injuries, rest days, travel distance
- Model should be recalibrated annually as NBA playing styles evolve

---

## AI Co-pilot Usage

AI tools (Claude — Anthropic) were used throughout this project and are fully documented in the **P.R.I.M.E. Log** page of the Streamlit dashboard and in Part 7 of the Jupyter notebook. All AI-generated code was independently tested; all AI-generated claims were source-verified. See the AI Methodology Appendix (PDF) for complete documentation.

---

## Citation

Lauga, N. (2020). *NBA Games Dataset* [Data set]. Kaggle.
https://www.kaggle.com/datasets/nathanlauga/nba-games
Accessed: April 21, 2026.
