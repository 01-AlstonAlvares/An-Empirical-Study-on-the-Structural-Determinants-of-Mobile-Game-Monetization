# From Mechanics to Microtransactions
### An Empirical Study on the Structural Determinants of Mobile Game Monetization

[![Python 3.11](https://img.shields.io/badge/Python-3.11%20%7C%203.14-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.2.0-orange.svg)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-0.51.0-green.svg)](https://shap.readthedocs.io/)
[![DiCE](https://img.shields.io/badge/DiCE-0.12-purple.svg)](https://github.com/interpretml/DiCE)
[![Tests](https://img.shields.io/badge/Tests-5%2F5%20Passing-brightgreen.svg)](tests/test_pipeline.py)
[![License](https://img.shields.io/badge/License-CC0%20%2F%20MIT-lightgrey.svg)](LICENSE)

An end-to-end Explainable AI (XAI) research project investigating how core mechanical design parameters determine monetization architecture in the mobile gaming industry. Built under rigorous academic standards evaluating explanation **faithfulness**, **stability**, **human decision utility**, **actionability**, and **ethical proxy auditing**.

---

## 1. Project Overview & Operational Context

During pre-production, mobile game studio leadership faces a critical technical fork in backend architecture: greenlighting an **Aggressive In-App Purchase (IAP) / Pay-to-Progress** model ($Y=1$) or adopting a lean, **Ad-Supported / Freemium** loop ($Y=0$).

* **Target Decision Maker:** **Maya**, Lead Producer at an independent mobile development studio.
* **The Decision:** Determining whether planned game mechanics organically justify the heavy fixed engineering overhead of an IAP economy (client-server authoritative state validation, server-side wallet ledgers, hardened anti-cheat systems, customer support, and continuous live-ops event tooling) before capital is deployed.
* **Consequence of Error:** Building heavy IAP infrastructure for a title whose mechanics fail to drive whale monetization risks studio insolvency. Conversely, under-monetizing a naturally spend-elastic title leaves critical studio runway uncaptured.

```
                  ┌─────────────────────────────────────────┐
                  │ Pre-Production Game Design Concept (Maya)│
                  └────────────────────┬────────────────────┘
                                       │
                     ┌─────────────────┴─────────────────┐
                     ▼                                   ▼
        ┌─────────────────────────┐         ┌─────────────────────────┐
        │  Aggressive IAP Loop    │         │ Ad-Supported Freemium   │
        │         (Y = 1)         │         │         (Y = 0)         │
        ├─────────────────────────┤         ├─────────────────────────┤
        │ • Max IAP >= $19.99     │         │ • Max IAP < $4.99       │
        │ • Hardened anti-cheat   │         │ • Contextual ad banners │
        │ • Authoritative servers │         │ • Lightweight backend   │
        │ • Live-ops pipelines    │         │ • Zero whale dependency │
        └─────────────────────────┘         └─────────────────────────┘
```

---

## 2. Dataset & Kaggle Data Access

The empirical analysis is conducted on store metadata spanning over 17,000 mobile titles under permissive open-access licenses.

> **Kaggle Dataset Download Links:**
> * **Google Play Store Apps: Data of 10k Play Store apps for analysing the Android market.**  
>   `https://www.kaggle.com/datasets/lava18/google-play-store-apps` 
> * **17K Mobile App Store Dataset:Every strategy game on the Apple App Store**  
>   `https://www.kaggle.com/datasets/tristan581/17k-apple-app-store-strategy-games` 
> * **Google Play Store Apps:Google Play Store App data of 2.3 Million+ applications.**  
>   `https://www.kaggle.com/datasets/gauthamp10/google-playstore-apps`

### Data Ingestion Setup
1. Download raw store metadata CSV files from the Kaggle links above.
2. Create a raw data directory in Data Directory.
3. Place the downloaded files into the repository's raw data directory:
   ```text
   Data/raw/appstore_games.csv
   ```
4. Run `notebooks/01_data_ingestion_and_labeling.ipynb` to parse heterogeneous storage formats into continuous megabytes, filter out 4,071 ambiguous mid-tier entries ($4.99–$19.99), and generate leakage-free stratified splits (`data/processed/train.parquet` and `data/processed/test.parquet`).

---

## 3. Repository Architecture

```
├── data/
│   ├── raw/                              # Kaggle App Store / Google Play raw CSVs
│   └── processed/                        # Cleaned train.parquet, test.parquet, feature metadata
├── notebooks/
│   ├── 01_data_ingestion_and_labeling.ipynb
│   ├── 02_exploratory_data_analysis.ipynb
│   ├── 03_feature_engineering_and_modeling.ipynb
│   ├── 04_faithfulness_evaluation.ipynb
│   ├── 05_stability_evaluation.ipynb
│   ├── 06_human_evidence_study.ipynb
│   └── 07_actionable_counterfactuals.ipynb
├── src/                                  # Helper utility modules imported by notebooks
│   ├── __init__.py
│   ├── config.py                         # Paths, random seeds (SEED=42), global constants
│   ├── data_helpers.py                   # Parsing and schema definitions
│   ├── metric_helpers.py                 # AUDC, Spearman rank, and Jaccard functions
│   └── plot_helpers.py                   # Standardized matplotlib styling functions
├── artifacts/
│   ├── figures/                          # Exported PNGs for the final written report
│   └── tables/                           # Exported Markdown and CSV results tables
├── tests/
│   └── test_pipeline.py                  # Integration checks ensuring zero split leakage
├── requirements.txt                      # Strict pinned pip dependencies
├── report.md                             # Comprehensive research and evaluation report
└── MODEL_CARD.md                         # Standardized model card per grading criteria
```

---

## 4. Empirical Visualizations & Model Explanations

### 4.1 Model Performance & Complexity Trade-Off (Criteria 2 & 7)

To justify using an opaque ensemble over a simple baseline, we trained three architectures across 93 encoded mechanical features on an identical held-out test split of 2,588 titles:

![Comparative ROC Curves](artifacts/figures/model_roc_curves.png)

| Model Architecture | Interpretability Status | Test Accuracy | Precision | Recall (Class 1) | Macro F1 | AUC-ROC | Log-Loss |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Decision Tree (depth=2)** | Transparent (3 splits) | 84.89% | 71.76% | 26.24% | 0.6491 | 0.7973 | 0.3715 |
| **Logistic Regression** | Linear Baseline (93 weights) | 87.06% | 73.90% | 43.23% | 0.7350 | 0.8464 | 0.3330 |
| **XGBoost (300 trees, depth 6)** | High Opacity (~18.9k splits) | **88.68%** | **75.75%** | **54.41%** | **0.7832** | **0.9033** | **0.2764** |

> **Model Explanation & Complexity Justification:**  
> A simple linear model achieves an AUC of 0.846 but misses over 56% of aggressive microtransaction titles. The complex XGBoost model is retained because it captures non-linear **multi-feature interaction thresholds**: client package footprint (`size_in_mb`) interacts compounding with network synchronicity (`is_synchronous_multiplayer`). While a large single-player game can succeed as an ad-supported freemium title, combining large asset sizes with real-time PvP matchmaking creates an exponential surge in microtransaction dependency. XGBoost captures this boundary naturally, boosting minority recall by 11.2 percentage points and reducing cross-entropy loss by 17.0%.

---

### 4.2 Faithfulness Quantification: Feature Deletion Curves (Criterion 3)

Faithfulness tests whether explanations mirror the model's true internal computational logic rather than presenting cosmetically appealing attributions. We executed a 13-step feature masking protocol on the test set:

![Faithfulness Deletion Curves](artifacts/figures/faithfulness_deletion_curve.png)

| Deletion Strategy | Baseline AUC ($k=0$) | Step 1 AUC ($k=1$) | Step 5 AUC ($k=5$) | Normalized AUDC Score | Faithfulness Verification |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **TreeSHAP-Guided** | 0.9033 | **0.8164** (-0.087) | **0.6841** (-0.219) | **0.5498** | **Highest Faithfulness** (Steepest drop) |
| **Random Control (20 Seeds)** | 0.9033 | 0.9015 (-0.002) | 0.8982 (-0.005) | 0.7524 | Baseline decay |
| **Inverse Deletion Control** | 0.9033 | 0.9033 (0.000) | 0.9033 (0.000) | 0.8319 | Uninformative tail |

> **Model Explanation & Empirical Meaning:**  
> When features are masked in order of TreeSHAP attribution, the model's test AUC suffers an immediate **cliff drop**: removing just the single top-ranked feature (`size_in_mb`) causes AUC to plummet from 0.9033 to 0.8164 (-0.087). By step $k=5$, AUC collapses to 0.6841. In contrast, 20-seed random masking retains an AUC of 0.8982 at $k=5$. TreeSHAP achieves an Area Under the Deletion Curve (AUDC) of **0.5498**, drastically lower than the random control (**0.7524**). This proves mathematically that TreeSHAP isolates the true computational drivers of the ensemble.

---

### 4.3 Explainer Stability Auditing (Criterion 4)

We audited explainer stability across 50 representative test game profiles under two operational stress tests:

![Explainer Stability Distributions](artifacts/figures/stability_distributions.png)

| Explainer & Evaluation Stress Test | Mean Spearman $r_s$ | Mean Top-5 Jaccard Index | Top-3 Feature Retention Rate | Production Operational Viability |
| :--- | :---: | :---: | :---: | :--- |
| **KernelSHAP (10 Seed Replications)** | $0.346 \pm 0.167$ | $0.338 \pm 0.179$ | N/A | **Unsuitable for Production** (High variance) |
| **TreeSHAP ($\pm 2\%$ Input Perturbation)** | $0.982 \pm 0.014$ | $0.941 \pm 0.052$ | **96.0%** (48/50 games stable) | **Mandated for Maya's Greenlight Reviews** |

> **Model Explanation & Policy for Maya:**  
> Monte Carlo sampling explainers (KernelSHAP, LIME) demonstrate extreme seed instability: feature attribution rankings shift substantially across runs ($r_s = 0.346$, Jaccard overlap = $0.338$), meaning identical design documents would yield conflicting explanations. In contrast, deterministic TreeSHAP maintains **96.0% top-3 feature stability** under $\pm 2\%$ input noise ($\Delta p < 0.01$). Studio policy mandates deterministic TreeSHAP for all production greenlight reviews.

---

### 4.4 Human Forward Simulation Study (Criterion 5)

We evaluated human prediction accuracy across $N = 10$ evaluators (indie game developers and peers) across 100 simulation trials:

![Human Evaluation Treatment Cards](artifacts/figures/human_sim_treatment_cards.png)

| Evaluation Cohort | Trial Count | Human Prediction Accuracy | Subjective Confidence (1–5 Scale) | Primary Cognitive Error Mode |
| :--- | :---: | :---: | :---: | :--- |
| **Control Group (Mechanics Alone)** | 50 trials | **62.0%** | $2.8 \pm 0.8$ | Overestimating monetization risk on large single-player titles |
| **Treatment Group (Mechanics + SHAP)** | 50 trials | **88.0%** | $4.2 \pm 0.6$ | Minimal (Misjudging borderline update cadence tags) |
| **Net Performance Lift** | -- | **+26.0 percentage points** | **+1.4 points** | Substantial reduction in developer cognitive bias |

> **Model Explanation & Evaluator Feedback:**  
> Providing visual TreeSHAP waterfall plots improved human decision accuracy from 62.0% to 88.0% (+26.0 percentage points). Evaluators noted: *"Without the SHAP waterfall, I assumed any 600 MB RPG was pay-to-progress, but the plot showed the absence of multiplayer pushed it firmly into freemium."* (Note: With $N=10$, we report this as descriptive decision support evidence without claiming formal statistical significance).

---

### 4.5 Actionable Counterfactual Recourse via Constrained DiCE (Criterion 6)

Using `dice-ml`, we locked all immutable constraints (`primary_genre`, `secondary_genre`, `minimum_os_version`, `content_rating`) and generated minimal mechanical pivots for 5 high-risk games ($p > 0.93$):

![Actionable Counterfactual Modifications](artifacts/figures/counterfactual_modifications.png)

| Candidate Game Profile | Original Probability | Counterfactual Probability | Core Mechanical Pivot Required | Feasibility & Production Impact |
| :--- | :---: | :---: | :--- | :--- |
| **Game A** (Action, 850 MB, Real-Time PvP) | $p = 0.988$ | **$p = 0.382$** | Compress footprint to 140 MB; switch PvP to Asynchronous Ghost Racing | **High:** Asset streaming reduces server validation overhead |
| **Game B** (Strategy, 420 MB, MMO Guilds) | $p = 0.993$ | **$p = 0.415$** | Decouple synchronous matchmaking; compress client to 115 MB | **High:** Eliminates stateful game server infrastructure |
| **Game C** (Role-Playing, 680 MB, Live Co-op) | $p = 0.981$ | **$p = 0.366$** | Transition live co-op to asynchronous companion hiring; stream audio | **Moderate:** Requires redesign of dungeon progression loop |
| **Game D** (Sports/Racing, 510 MB, Real-Time) | $p = 0.974$ | **$p = 0.442$** | Switch to asynchronous ghost matchmaking; reduce base download to 135 MB | **High:** Utilizes turn-based leaderboard backend |
| **Game E** (Adventure, 390 MB, Real-Time PvP) | $p = 0.965$ | **$p = 0.395$** | Decouple multiplayer; implement episodic dynamic asset downloads | **High:** Drastically reduces Day-1 download friction |

> **Actionable Takeaways for Maya:**  
> 1. **Asset Streaming:** Compressing initial download packages below 150 MB via on-demand CDN delivery keeps titles within casual freemium discovery funnels.  
> 2. **Multiplayer Decoupling:** Decoupling real-time synchronous PvP to asynchronous leaderboards or ghost racing drops model-predicted monetization risk by over 50 percentage points, eliminating the need for expensive anti-cheat and live-ops infrastructure.

---

### 4.6 Statutory Ethical Proxy Audit (Criterion 8)

We conducted a statutory proxy audit to examine whether mechanical attributes proxy for protected classes under youth protection regulations (COPPA, FTC dark pattern guidelines):

![Spearman Rank Correlations](artifacts/figures/spearman_correlations.png)

| Sensitive Demographic Proxy | Correlation with Monetization ($Y$) | Correlation with Real-Time Multiplayer | Statutory Regulatory Implication for Maya |
| :--- | :---: | :---: | :--- |
| **Youth Proxy (`4+`, `9+` Ratings)** | **$r_s = -0.162$** ($p < 0.001$) | **$r_s = -0.141$** | Child-targeted games default to freemium. Maya must strictly prohibit behavioral ad SDKs to prevent COPPA violations. |
| **Mature Proxy (`17+` Rating)** | **$r_s = +0.108$** ($p < 0.001$) | **$r_s = +0.076$** | Mature classifications correlate with spend-elastic adult demographics capable of credit card transactions. |

---

## 5. Step-by-Step Research Pipeline

| Notebook | Phase | Objective & Methodology | Primary Outputs |
| :--- | :--- | :--- | :--- |
| **`01_data_ingestion_and_labeling.ipynb`** | Data Ingestion & Boundary Cutoffs | Ingests 17,007 raw titles; parses sizes into continuous MB; derives ground truths ($Y=1$: IAP $\ge \$19.99$; $Y=0$: ads + IAP $< \$4.99$); drops 4,071 ambiguous mid-tier entries; exports stratified 80/20 train/test splits. | `data/processed/train.parquet`<br>`data/processed/test.parquet`<br>`data_filtering_summary.csv` |
| **`02_exploratory_data_analysis.ipynb`** | EDA & Statutory Proxy Audit | Audits continuous skewness (`size_in_mb` skewness = 3.61); evaluates Spearman rank collinearity; conducts statutory minor proxy audit (`4+`/`9+` ratings correlate at $r_s = -0.162$ with aggressive IAP). | `eda_distributions.png`<br>`spearman_correlations.png`<br>`ethical_proxy_audit.csv` |
| **`03_feature_engineering_and_modeling.ipynb`** | Modeling & Complexity Evidence | Segregates 4 `IMMUTABLE_FEATURES` from 5 `MUTABLE_FEATURES`; trains Decision Tree (AUC=0.797) and Logistic Regression (AUC=0.846) baselines alongside complex opaque XGBoost (300 trees, depth 6, AUC=0.903, Log-Loss=0.276). | `model_roc_curves.png`<br>`model_performance_comparison.csv`<br>`artifacts/models/` |
| **`04_faithfulness_evaluation.ipynb`** | Faithfulness via Deletion Curves | Tests whether TreeSHAP mirrors true computational logic via 13-step feature masking against 20-seed Random Deletion and Inverse Deletion controls. Quantifies AUDC (TreeSHAP 0.5498 vs Random 0.7524). | `faithfulness_deletion_curve.png`<br>`faithfulness_audc_summary.csv` |
| **`05_stability_evaluation.ipynb`** | Explainer Stability Auditing | Assesses explainer variance across 50 test profiles: Test A demonstrates KernelSHAP sampling instability ($r_s = 0.346$); Test B proves TreeSHAP maintains 96.0% top-3 stability under continuous $\pm 2\%$ perturbation. | `stability_distributions.png`<br>`stability_evaluation_summary.csv` |
| **`06_human_evidence_study.ipynb`** | Human Forward Simulation Study | Forward simulation protocol on $N=10$ evaluators across 100 trials: prediction accuracy surges from 62.0% to 88.0% (+26.0% lift) with SHAP waterfall plots; logs qualitative participant feedback. | `human_sim_treatment_cards.png`<br>`human_simulation_packet.md`<br>`human_simulation_results.csv` |
| **`07_actionable_counterfactuals.ipynb`** | Actionable Recourse via DiCE | Constrains DiCE optimization strictly to mutable mechanics while freezing immutable genre/platform constraints; flips 5 high-risk games ($p > 0.93 \rightarrow p < 0.45$) via asset streaming (<150 MB) and PvP decoupling. | `counterfactual_modifications.png`<br>`actionable_counterfactuals_summary.csv` |

---

## 6. Installation & Quickstart

### Prerequisites
* Python 3.11 or Python 3.14
* Git

### Setup Instructions
```bash
# 1. Clone repository
git clone https://github.com/your-username/mobile-game-monetization-xai.git
cd mobile-game-monetization-xai

# 2. Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# 3. Install strictly pinned dependencies
pip install -r requirements.txt

# 4. Run automated test suite
pytest tests/test_pipeline.py -v
```

---

## 7. Documentation & Key Deliverables

* **Academic Research Report:** [`report.md`](report.md) — Comprehensive 4-page research report detailing executive takeaways, empirical proofs, and production directives.
* **Standardized Model Card:** [`MODEL_CARD.md`](MODEL_CARD.md) — Detailed specifications covering intended user (Maya), out-of-scope uses, opacity parameters, quantitative XAI audits, and ethical proxy guidelines.
* **Integration Tests:** [`tests/test_pipeline.py`](tests/test_pipeline.py) — 5 automated test cases asserting zero split leakage, schema compliance, and artifact persistence.

---

## 8. Citation & License

This project is licensed under the MIT License and uses public domain data licensed under CC0.

```bibtex
@article{xds_mobile_monetization_2026,
  title={From Mechanics to Microtransactions: An Empirical Study on the Structural Determinants of Mobile Game Monetization},
  author={Alston Anthony Alvares and Muhammad Fahad Waqar and Biki Nath Newa and Tommesh Sharad Commar},
  institution={Asian Institute of Technology},
  year={2026}
}
```
