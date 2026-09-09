# Model Card: Structural Predictors of Mobile Strategy Game Monetization Archetypes

## 1. Model Details

### 1.1 Overview
- **Project Title:** From Mechanics to Microtransactions: An Empirical Study on Structural Predictors of Mobile Strategy Game Monetization Archetypes
- **Primary Model Architecture:** Extreme Gradient Boosted Trees (`XGBoostClassifier`)
- **Version:** 1.0.0 (Production Release)
- **Model Framework:** XGBoost 3.2.0 / Scikit-Learn 1.8.0
- **Explainability Stack:** TreeSHAP (SHAP 0.51.0) and Diverse Counterfactual Explanations (DiCE 0.12)
- **Primary Artifact Location:** `artifacts/models/xgboost_model.joblib`

### 1.2 Intended User & Decision (Persona: Maya)
- **Primary Intended User:** Maya, a Lead Producer at an independent mid-sized mobile game studio.
- **Target Decision Context:** Deciding whether to greenlight the heavy backend architectural investment required for an in-app purchase (IAP) microtransaction economy or to pivot the title's core loop toward a lean, ad-supported freemium monetization loop during pre-production.
- **Operational Value:** Provides quantitative diagnostic attributions (TreeSHAP) and prescriptive recourse (DiCE) showing which specific planned game mechanics associate with the pay-to-progress archetype, allowing Maya to optimize technical architecture before capital is deployed.

### 1.3 Target Operationalization & Out-of-Scope Uses
- **Target Operationalization:** The model predicts a **heuristically defined monetization archetype** operationalized via public storefront catalog metadata ($Y=1$: catalog contains IAP $\ge \$19.99$; $Y=0$: ad indicators + IAP $< \$4.99$).
- **Direct Financial Performance Forecasting (Out-of-Scope):** Not designed to predict actual commercial performance metrics, including financial revenue, ARPU, ARPPU, player conversion rates, or lifetime value (LTV).
- **Dynamic In-Game Pricing (Out-of-Scope):** Prohibited from adjusting live microtransaction price tiers dynamically based on player behavior or perceived elasticity.
- **Post-Launch Telemetry Optimization (Out-of-Scope):** Designed strictly for pre-production architectural decisions; not validated on post-launch live-ops telemetry.
- **Predatory Dark Pattern Optimization (Prohibited):** Strictly prohibited from optimizing mechanic configurations to induce addictive or coercive spending behaviors, especially in titles with family/minor audiences.

---

## 2. Model Performance & Complexity Specifications

### 2.1 Model Complexity & Opacity Parameters (Criterion 7)
To evaluate the opacity trade-off, the complex ensemble model is parameterized with explicit structural constraints:
- **Base Estimators:** Exactly `n_estimators = 300` boosting iterations.
- **Maximum Tree Depth:** Exactly `max_depth = 6` levels per tree.
- **Learning Rate:** `learning_rate = 0.05` with log-loss optimization (`eval_metric = "logloss"`).
- **Encoded Feature Space:** 93 model-input features generated from nine engineered structural attributes (via one-hot encoding and standardization, exceeding the 40+ criterion requirement).
- **Ensemble Opacity:** Encapsulates up to $300 \times (2^6 - 1) = 18,900$ potential decision splits, representing an opaque, non-linear predictive surface.
- **Deterministic Seed:** Frozen across all stages with `random_state = 42`.

### 2.2 Side-by-Side Performance Comparison (Criterion 2 vs. Criterion 7)
Evaluated on the identical held-out test split of 2,588 titles (80/20 stratified split, zero split leakage):

| Model Family | Opacity Classification | Accuracy | Precision (Class 1) | Recall (Class 1) | Macro F1 | AUC-ROC | Log-Loss |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Decision Tree (depth=2)** | Transparent (3 splits) | 84.89% | 71.76% | 26.24% | 0.6491 | 0.7973 | 0.3715 |
| **Logistic Regression** | Linear (93 weights) | 87.06% | 73.90% | 43.23% | 0.7350 | 0.8464 | 0.3330 |
| **XGBoost (300 trees, depth 6)** | High Opacity (~18.9k splits) | **88.68%** | **75.75%** | **54.41%** | **0.7832** | **0.9033** | **0.2764** |

### 2.3 Evaluation on Imbalanced Classes & Complexity Justification
Because Class 0 comprises 82% of instances, raw accuracy is an incomplete metric (a naive majority-class classifier achieves 82%). Evaluation centers on **AUC-ROC (0.903)**, **Macro-F1 (0.783)**, and **Class-1 Recall (54.4%)**.
While the linear baseline achieves an AUC-ROC of 0.846, it captures only 43.2% of aggressive microtransaction titles. The complex XGBoost model achieves an AUC-ROC of 0.903 and increases recall on minority microtransaction titles to 54.4% (at 75.7% precision) while reducing log-loss from 0.333 down to 0.276.
This difference stems from non-linear **multi-feature interaction associations**: client download footprints interact non-linearly with synchronous multiplayer networking. Large games without multiplayer frequently remain viable as single-player freemium titles, but large download footprints combined with real-time PvP matchmaking associate strongly with microtransaction dependency. Tree ensembles capture these interaction thresholds naturally without manual polynomial expansion.

---

## 3. Explainability, Faithfulness & Stability Summary

### 3.1 Faithfulness Evidence (Criterion 3: Feature Deletion Curves)
Faithfulness was evaluated via iterative feature masking on the held-out test set, comparing TreeSHAP ranking against Random and Inverse deletion controls across 13 deletion budgets ($k \in [0, \dots, 93]$):
- **TreeSHAP Deletion (Top Features First):** Produces an immediate cliff drop. Masking just the single top-ranked feature (`size_in_mb`) causes AUC to decline from **0.903** to **0.816** (-0.087). Masking 5 features drops AUC to **0.684** (-0.219).
- **Random Deletion Control (20 Seeds):** Retains an AUC of **0.898** after 5 features are masked, exhibiting slow, gradual decay.
- **Inverse Deletion Control (Bottom Features First):** Retains an AUC of **0.903** even after 40 features are masked.
- **Area Under Deletion Curve (AUDC):**
  - TreeSHAP AUDC: **0.5498** (Lowest area indicates highest faithfulness)
  - Random Deletion AUDC: **0.7524**
  - Inverse Deletion AUDC: **0.8319**
- **Interpretation:** These results provide empirical evidence that the highest-ranked TreeSHAP features correspond closely to model inputs that materially affect predictive scoring.

### 3.2 Stability Evaluation (Criterion 4: Seeds & Perturbations)
Evaluated across a fixed cohort of 50 representative test profiles:
- **Test A (Seed Variation on Sampling Explainers):** Running KernelSHAP across 10 random seeds revealed significant instability due to Monte Carlo sampling variance: mean pairwise Spearman rank correlation was only $r_s = 0.346$ ($\pm 0.167$) and mean top-5 Jaccard overlap was only $0.338$ ($\pm 0.179$). This demonstrates that sampling explainers are unsuitable for high-stakes production reviews.
- **Test B (Small Input Epsilon Perturbations on TreeSHAP):** Continuous features were perturbed by $\pm 2\%$, strictly bounded to ensure model output probability changed by $\Delta p < 0.01$ (mean observed $\Delta p = 0.0054$). Under TreeSHAP, **96.0%** of instances maintained identical top-3 feature sets (only 2 out of 50 games exhibited a boundary flip). TreeSHAP is mathematically deterministic, and our perturbation experiment indicates high attribution stability under tested input noise.

### 3.3 Pilot Human Decision Support (Criterion 5: Forward Simulation Study)
Evaluated with $N = 10$ human participants (graduate classmates and mobile game developers). Each evaluator assessed all 10 standardized game cards (5 Control, 5 Treatment), producing $10 \times 10 = 100$ total evaluator-card trials:
- **Design:** Within-subject evaluation comparing 5 Control cards (mechanics alone) against 5 Treatment cards (mechanics + local TreeSHAP waterfall plots).
- **Control Group (Mechanics Alone):** Prediction accuracy was **62.0%** (confidence 2.8/5.0). Evaluators routinely misclassified large single-player games.
- **Treatment Group (Mechanics + SHAP):** Prediction accuracy improved to **88.0%** (+26.0 percentage points, confidence 4.2/5.0).
- **Sample Scale Notice:** We explicitly acknowledge sample size constraints ($N=10$) and present these findings as pilot descriptive evidence without claiming formal statistical significance.

### 3.4 Model-Derived Actionable Recourse (Criterion 6: Constrained DiCE Optimization)
- **Constraint Enforcement:** All immutable features (`primary_genre`, `secondary_genre`, `minimum_os_version`, `content_rating`) were strictly locked.
- **Recourse Generation:** For 5 high-risk games ($p \in [0.988, 0.993]$), DiCE generated minimal feature deltas associated with flipping predictions to the Ad-Supported freemium archetype ($p \in [0.366, 0.489]$).
- **Model-Derived Levers for Maya:**
  1. Asset Footprint Compression: Reducing download size below 150 MB via on-demand dynamic asset streaming.
  2. Multiplayer Architecture Decoupling: Swapping real-time synchronous PvP matchmaking for asynchronous leaderboards or turn-based ghost matchmaking.

---

## 4. Ethical & Content-Rating Regulatory Proxy Audit (Criterion 8)

### 4.1 Content-Rating Regulatory Proxy Identification
Storefront metadata does not record player demographic profiles. To prevent regulatory non-compliance (COPPA, FTC dark patterns), content ratings were audited as statutory proxies under youth protection frameworks:
- `proxy_family_youth_under_12`: `content_rating` $\in$ `['4+', '9+']` (Youth / Family classifications)
- `proxy_mature_17_plus`: `content_rating` == `'17+'` (Adult / Mature classifications)

*Demographic Clarification:* App Store age ratings indicate **content suitability classifications** and should not be interpreted as direct measurements of player demographics (e.g., a `4+` rating indicates content suitability for all ages, not that only children play the game).

### 4.2 Empirical Correlation Audit
Exact correlation coefficients with structural features and archetype target labels:
- **Youth Proxy vs. Monetization Archetype ($Y$):** $r_s = -0.162$ ($p < 0.001$). Lower content-age classifications are negatively associated with the aggressive-IAP archetype.
- **Youth Proxy vs. Synchronous Multiplayer:** $r_s = -0.141$. Real-time PvP is concentrated in older age tiers due to toxicity and chat moderation requirements.
- **Mature Proxy vs. Monetization Archetype ($Y$):** $r_s = +0.108$ ($p < 0.001$). Mature content classifications are positively associated with the aggressive-IAP archetype.

### 4.3 Production Risk Mitigation for Maya
- **Regulatory Warning:** Because lower content-age classifications associate empirically with ad-supported freemium loops, studios often deploy ad monetization. Maya should require a privacy/legal review of advertising SDKs and child-directed data practices before deployment to ensure regulatory compliance (e.g., COPPA, FTC dark pattern guidelines).
- **Engineering Recommendation:** For any game rated `4+` or `9+`, Maya should bypass behavioral ad tracking, implementing contextual or premium monetization loops instead.

---

## 5. Dataset & Feature Schema

### 5.1 Dataset Lineage, Candidate Sources & Scoping
- **Candidate Sources Evaluated in `data/raw/`:**
  1. *Primary Modeling Cohort:* 17K Apple App Store Strategy Games (`appstore_games.csv`, 17,007 titles; CC0 license).
  2. *Google Play 10K Storefront Dataset:* (`googleplaystore.csv`, 10,841 records by lava18; CC0 license).
  3. *Google Play 2.3M Storefront Dataset:* (`Google-Playstore.csv`, 2,312,944 records by gauthamp10; CC0 license).
- **Primary Scoping Rationale:** Only `appstore_games.csv` records granular in-app purchase price arrays (`$0.99` to `$99.99`) required to mathematically operationalize the monetization target ($Max\_IAP \ge \$19.99$ vs $< \$4.99$). Furthermore, over 85% of Google Play records are non-gaming utility apps, and cross-store pooling introduces platform packaging confounders.
- **Ambiguous Mid-Tier Games Dropped:** 4,071 titles (games with $\$4.99 \le \text{Max IAP} < \$19.99$ or upfront paid non-ad games)
- **Final Clean Analyzed Cohort:** 12,936 titles (10,610 Ad-Supported, 2,326 Aggressive IAP)

### 5.2 Feature Schema
- **Immutable Features (Frozen):**
  - `primary_genre` (Categorical)
  - `secondary_genre` (Categorical)
  - `minimum_os_version` (Categorical)
  - `content_rating` (Categorical: 4+, 9+, 12+, 17+)
- **Mutable Features (Actionable):**
  - `size_in_mb` (Continuous float: package download size)
  - `is_synchronous_multiplayer` (Binary: real-time PvP/multiplayer)
  - `session_pacing_tag` (Categorical: Turn-Based, Real-Time, Hypercasual/Idle, Session-Based)
  - `supported_languages_count` (Integer: localization breadth)
  - `days_since_last_update` (Continuous integer: live-ops staleness)
- **Target Variable ($Y$):**
  - $Y = 1$ (**Aggressive IAP Archetype**): `has_iap == True` AND `max_iap_price >= $19.99`
  - $Y = 0$ (**Ad-Supported / Freemium Archetype**): `contains_ads == True` AND `max_iap_price < $4.99`
