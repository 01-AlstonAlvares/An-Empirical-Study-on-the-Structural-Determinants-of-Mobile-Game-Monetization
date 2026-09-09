# Model Card: Structural Determinants of Mobile Game Monetization

## 1. Model Details

### 1.1 Overview
- **Project Title:** From Mechanics to Microtransactions: An Empirical Study on the Structural Determinants of Mobile Game Monetization
- **Primary Model Architecture:** Extreme Gradient Boosted Trees (`XGBoostClassifier`)
- **Version:** 1.0.0 (Production Release)
- **Model Framework:** XGBoost 3.2.0 / Scikit-Learn 1.8.0
- **Explainability Stack:** TreeSHAP (SHAP 0.51.0) and Diverse Counterfactual Explanations (DiCE 0.12)
- **Primary Artifact Location:** `artifacts/models/xgboost_model.joblib`

### 1.2 Intended User & Decision (Persona: Maya)
- **Primary Intended User:** Maya, a Lead Producer at an independent mid-sized mobile game studio.
- **Target Decision Context:** Deciding whether to greenlight the heavy backend architectural investment required for an in-app purchase (IAP) microtransaction economy or to pivot the title's core loop toward a lean, ad-supported freemium monetization loop during pre-production.
- **Operational Value:** Provides quantitative diagnostic attributions (TreeSHAP) and prescriptive recourse (DiCE) showing which specific planned game mechanics push the title into a pay-to-progress archetype, allowing Maya to optimize technical architecture before capital is burned.

### 1.3 Out-of-Scope & Prohibited Uses
- **Dynamic In-Game Pricing:** Prohibited from adjusting live microtransaction price tiers dynamically based on player behavior or perceived elasticity.
- **Individual Player Lifetime Value (LTV) Forecasting:** Not designed to predict financial revenue per user, ARPU, or retention cohorts.
- **Post-Launch Telemetry Optimization:** Designed strictly for pre-production architectural decisions; not validated on post-launch live-ops telemetry.
- **Predatory Dark Pattern Optimization:** Strictly prohibited from optimizing mechanic configurations to induce addictive or coercive spending behaviors, especially in titles with family/minor audiences.

---

## 2. Model Performance & Opacity Specifications

### 2.1 Model Complexity & Opacity Parameters (Criterion 7)
To evaluate the opacity trade-off, the complex ensemble model is parameterized with explicit structural constraints:
- **Base Estimators:** Exactly `n_estimators = 300` boosting iterations.
- **Maximum Tree Depth:** Exactly `max_depth = 6` levels per tree.
- **Learning Rate:** `learning_rate = 0.05` with log-loss optimization (`eval_metric = "logloss"`).
- **Encoded Feature Space:** 93 one-hot and scaled mechanical features (expanded from 9 structural attributes, well exceeding the 40+ criterion requirement).
- **Ensemble Opacity:** Encapsulates up to $300 \times (2^6 - 1) = 18,900$ potential decision splits, representing an opaque, non-linear predictive surface.
- **Deterministic Seed:** Frozen across all stages with `random_state = 42`.

### 2.2 Side-by-Side Performance Comparison (Criterion 2 vs. Criterion 7)
Evaluated on the identical held-out test split of 2,588 titles (80/20 stratified split, zero split leakage):

| Model Family | Opacity Classification | Accuracy | Precision | Recall (Class 1) | Macro F1 | AUC-ROC | Log-Loss |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Decision Tree (depth=2)** | Transparent (3 splits) | 0.8489 | 0.7176 | 0.2624 | 0.6491 | 0.7973 | 0.3715 |
| **Logistic Regression** | Linear (93 weights) | 0.8706 | 0.7390 | 0.4323 | 0.7350 | 0.8464 | 0.3330 |
| **XGBoost (300 trees, depth 6)** | High Opacity (~18.9k splits) | **0.8868** | **0.7575** | **0.5441** | **0.7832** | **0.9033** | **0.2764** |

### 2.3 Empirical Justification for Model Complexity
While the linear baseline achieves a respectable AUC-ROC of 0.846, it captures only 43.2% of aggressive microtransaction titles. The complex XGBoost model achieves an AUC-ROC of 0.903 and increases recall on minority microtransaction titles to 54.4% (at 75.7% precision) while reducing log-loss from 0.333 down to 0.276.
This difference stems from **multi-feature interaction thresholds**: client download footprints interact non-linearly with synchronous multiplayer networking. Large games without multiplayer are viable as single-player freemium titles, but large games with real-time PvP create a severe non-linear jump toward microtransaction dependency. Tree ensembles capture these interactions naturally without manual polynomial expansion.

---

## 3. Explainability, Faithfulness & Stability Summary

### 3.1 Faithfulness Evidence (Criterion 3: Feature Deletion Curves)
Faithfulness was evaluated via iterative feature masking on the held-out test set, comparing TreeSHAP ranking against Random and Inverse deletion controls across 13 deletion budgets ($k \in [0, \dots, 93]$):
- **TreeSHAP Deletion (Top Features First):** Produces an immediate cliff drop. Masking just the single top-ranked feature (`size_in_mb`) causes AUC to plummet from **0.903** to **0.816** (-0.087). Masking 5 features drops AUC to **0.684** (-0.219).
- **Random Deletion Control (20 Seeds):** Retains an AUC of **0.898** after 5 features are masked, exhibiting slow, gradual decay.
- **Inverse Deletion Control (Bottom Features First):** Retains an AUC of **0.903** even after 40 features are masked.
- **Area Under Deletion Curve (AUDC):**
  - TreeSHAP AUDC: **0.5498** (Lowest area proves highest faithfulness)
  - Random Deletion AUDC: **0.7524**
  - Inverse Deletion AUDC: **0.8319**

### 3.2 Stability Evaluation (Criterion 4: Seeds & Perturbations)
Evaluated across a fixed cohort of 50 representative test profiles:
- **Test A (Seed Variation on Sampling Explainers):** Running KernelSHAP across 10 random seeds revealed significant instability due to Monte Carlo sampling variance: mean pairwise Spearman rank correlation was only $r_s = 0.346$ ($\pm 0.167$) and mean top-5 Jaccard overlap was only $0.338$ ($\pm 0.179$). This proves that sampling explainers are unsuitable for high-stakes production reviews.
- **Test B (Small Input Epsilon Perturbations on TreeSHAP):** Continuous features were perturbed by $\pm 2\%$, strictly bounded to ensure model output probability changed by $\Delta p < 0.01$ (mean observed $\Delta p = 0.0054$). Under TreeSHAP, **96.0%** of instances maintained identical top-3 feature sets (only 2 out of 50 games exhibited a boundary flip). TreeSHAP is mathematically deterministic and highly robust to input noise.

### 3.3 Human Decision Support (Criterion 5: Forward Simulation Study)
Evaluated with $N = 10$ human participants (classmates and mobile game developers) across 100 total simulation trials:
- **Control Group (Mechanics Alone):** Prediction accuracy was **62.0%** (confidence 2.8/5.0). Evaluators routinely misclassified high-size single-player games.
- **Treatment Group (Mechanics + SHAP):** Prediction accuracy improved to **88.0%** (+26.0 percentage points, confidence 4.2/5.0).
- **Sample Size Constraint:** We explicitly acknowledge sample size limitations ($N=10$) and do not assert formal statistical significance; results indicate descriptive decision-support utility.

### 3.4 Actionable Recourse (Criterion 6: Constrained DiCE Optimization)
- **Constraint Enforcement:** All immutable features (`primary_genre`, `secondary_genre`, `minimum_os_version`, `content_rating`) were strictly locked.
- **Recourse Generation:** For 5 high-risk games ($p \in [0.988, 0.993]$), DiCE generated minimal feature deltas that flipped predictions to Ad-Supported freemium ($p \in [0.366, 0.489]$).
- **Key Levers for Maya:**
  1. Asset Footprint Compression: Reducing download size below 150 MB via on-demand asset pack delivery.
  2. Multiplayer Architecture Decoupling: Swapping real-time synchronous PvP matchmaking for asynchronous leaderboards or turn-based ghost matchmaking.

---

## 4. Ethical Proxy Audit & Responsible AI Considerations (Criterion 8)

### 4.1 Sensitive Proxy Identification
Store metadata does not contain player demographics. To prevent regulatory non-compliance (COPPA, FTC dark patterns), content ratings were audited as statutory proxies for children and minors:
- `proxy_family_youth_under_12`: `content_rating` $\in$ `['4+', '9+']` (Youth / Family classifications)
- `proxy_mature_17_plus`: `content_rating` == `'17+'` (Adult / Mature classifications)

### 4.2 Empirical Correlation Audit
Exact correlation coefficients with structural features and target labels:
- **Youth Proxy vs. Monetization Target ($Y$):** $r_s = -0.162$ ($p < 0.001$). Minor-targeted games correlate moderately with the ad-supported freemium archetype.
- **Youth Proxy vs. Synchronous Multiplayer:** $r_s = -0.141$. Real-time PvP is concentrated in older age tiers due to toxicity and chat moderation requirements.
- **Mature Proxy vs. Monetization Target ($Y$):** $r_s = +0.108$. Aggressive microtransactions correlate with adult demographics capable of direct credit card transactions.

### 4.3 Production Risk Mitigation for Maya
- **Regulatory Warning:** Because child-rated games are predicted as ad-supported, studios often deploy aggressive interstitial/rewarded video ad SDKs. This poses severe regulatory risks under COPPA if third-party ad networks track child device IDs.
- **Engineering Recommendation:** For any game rated `4+` or `9+`, Maya should bypass behavioral ad tracking, implementing contextual or premium monetization loops instead.

---

## 5. Dataset & Feature Schema

### 5.1 Dataset Lineage & Licensing
- **Source:** Kaggle App Store & Google Play Benchmark Metadata
- **License:** CC0: Public Domain / Permissive Open Access
- **Raw Rows Ingested:** 17,007 games
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
  - $Y = 1$ (**Aggressive IAP / Pay-to-Progress**): `has_iap == True` AND `max_iap_price >= $19.99`
  - $Y = 0$ (**Ad-Supported / Freemium**): `contains_ads == True` AND `max_iap_price < $4.99`
