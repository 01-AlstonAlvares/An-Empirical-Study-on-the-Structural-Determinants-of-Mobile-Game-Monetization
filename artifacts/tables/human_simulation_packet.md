# Mobile Game Monetization: Human Forward Simulation Evaluation Packet

**Participant Objective:** For each game profile card below, predict the machine learning model's classification:
- **Class 0:** Ad-Supported / Minimal IAP (Casual freemium loop)
- **Class 1:** Aggressive IAP / Pay-to-Progress (Heavy live-ops microtransactions)

---

## Part 1: Control Cards (Structural Mechanics Alone)
Evaluate each game based exclusively on its planned technical and design parameters.

### CARD-C1: Action Game Pitch
- **Primary Genre:** Action (Secondary: Entertainment)
- **Client Download Size:** 50.26 MB
- **Network Architecture:** Single-Player / Asynchronous
- **Session Pacing:** Session-Based
- **Target Content Rating:** 4+ (Minimum OS: iOS 10.0)
- **Localization Scope:** 1 languages
- **Days Since Last Update:** 1143 days

**Your Prediction:** [ ] Class 0 (Ad-Supported)  |  [ ] Class 1 (Aggressive IAP)
**Confidence (1-5):** ____
**Key Rationale:** ____________________________________________________

---

### CARD-C2: Action Game Pitch
- **Primary Genre:** Action (Secondary: Strategy)
- **Client Download Size:** 527.72 MB
- **Network Architecture:** Single-Player / Asynchronous
- **Session Pacing:** Real-Time
- **Target Content Rating:** 17+ (Minimum OS: iOS 11.0)
- **Localization Scope:** 1 languages
- **Days Since Last Update:** 98 days

**Your Prediction:** [ ] Class 0 (Ad-Supported)  |  [ ] Class 1 (Aggressive IAP)
**Confidence (1-5):** ____
**Key Rationale:** ____________________________________________________

---

### CARD-C3: Strategy Game Pitch
- **Primary Genre:** Strategy (Secondary: Casual)
- **Client Download Size:** 14.62 MB
- **Network Architecture:** Single-Player / Asynchronous
- **Session Pacing:** Session-Based
- **Target Content Rating:** 4+ (Minimum OS: iOS 9.0)
- **Localization Scope:** 1 languages
- **Days Since Last Update:** 1979 days

**Your Prediction:** [ ] Class 0 (Ad-Supported)  |  [ ] Class 1 (Aggressive IAP)
**Confidence (1-5):** ____
**Key Rationale:** ____________________________________________________

---

### CARD-C4: Role Playing Game Pitch
- **Primary Genre:** Role Playing (Secondary: Strategy)
- **Client Download Size:** 159.89 MB
- **Network Architecture:** Synchronous Real-Time Multiplayer
- **Session Pacing:** Session-Based
- **Target Content Rating:** 9+ (Minimum OS: iOS 10.0)
- **Localization Scope:** 1 languages
- **Days Since Last Update:** 465 days

**Your Prediction:** [ ] Class 0 (Ad-Supported)  |  [ ] Class 1 (Aggressive IAP)
**Confidence (1-5):** ____
**Key Rationale:** ____________________________________________________

---

### CARD-C5: Puzzle Game Pitch
- **Primary Genre:** Puzzle (Secondary: Strategy)
- **Client Download Size:** 41.53 MB
- **Network Architecture:** Single-Player / Asynchronous
- **Session Pacing:** Turn-Based
- **Target Content Rating:** 9+ (Minimum OS: iOS 11.0)
- **Localization Scope:** 25 languages
- **Days Since Last Update:** 343 days

**Your Prediction:** [ ] Class 0 (Ad-Supported)  |  [ ] Class 1 (Aggressive IAP)
**Confidence (1-5):** ____
**Key Rationale:** ____________________________________________________

---

## Part 2: Treatment Cards (Structural Mechanics + Local TreeSHAP Explanation)
Evaluate each game with the assistance of its local SHAP attribution plot (refer to artifacts/figures/human_sim_treatment_cards.png).

### CARD-T1: Casual Game Pitch
- **Primary Genre:** Casual (Secondary: Strategy)
- **Client Download Size:** 24.27 MB
- **Network Architecture:** Single-Player / Asynchronous
- **Session Pacing:** Real-Time
- **Target Content Rating:** 4+ (Minimum OS: iOS 11.0)
- **Localization Scope:** 1 languages
- **Days Since Last Update:** 549 days
- **SHAP Plot Reference:** Panel 1 in `artifacts/figures/human_sim_treatment_cards.png`

**Your Prediction:** [ ] Class 0 (Ad-Supported)  |  [ ] Class 1 (Aggressive IAP)
**Confidence (1-5):** ____
**Key Rationale:** ____________________________________________________

---

### CARD-T2: Entertainment Game Pitch
- **Primary Genre:** Entertainment (Secondary: Simulation)
- **Client Download Size:** 25.71 MB
- **Network Architecture:** Single-Player / Asynchronous
- **Session Pacing:** Session-Based
- **Target Content Rating:** 4+ (Minimum OS: iOS 11.0)
- **Localization Scope:** 1 languages
- **Days Since Last Update:** 167 days
- **SHAP Plot Reference:** Panel 2 in `artifacts/figures/human_sim_treatment_cards.png`

**Your Prediction:** [ ] Class 0 (Ad-Supported)  |  [ ] Class 1 (Aggressive IAP)
**Confidence (1-5):** ____
**Key Rationale:** ____________________________________________________

---

### CARD-T3: Strategy Game Pitch
- **Primary Genre:** Strategy (Secondary: Puzzle)
- **Client Download Size:** 66.21 MB
- **Network Architecture:** Single-Player / Asynchronous
- **Session Pacing:** Turn-Based
- **Target Content Rating:** 4+ (Minimum OS: iOS 10.0)
- **Localization Scope:** 4 languages
- **Days Since Last Update:** 207 days
- **SHAP Plot Reference:** Panel 3 in `artifacts/figures/human_sim_treatment_cards.png`

**Your Prediction:** [ ] Class 0 (Ad-Supported)  |  [ ] Class 1 (Aggressive IAP)
**Confidence (1-5):** ____
**Key Rationale:** ____________________________________________________

---

### CARD-T4: Strategy Game Pitch
- **Primary Genre:** Strategy (Secondary: Action)
- **Client Download Size:** 212.78 MB
- **Network Architecture:** Synchronous Real-Time Multiplayer
- **Session Pacing:** Session-Based
- **Target Content Rating:** 9+ (Minimum OS: iOS 10.0)
- **Localization Scope:** 1 languages
- **Days Since Last Update:** 841 days
- **SHAP Plot Reference:** Panel 4 in `artifacts/figures/human_sim_treatment_cards.png`

**Your Prediction:** [ ] Class 0 (Ad-Supported)  |  [ ] Class 1 (Aggressive IAP)
**Confidence (1-5):** ____
**Key Rationale:** ____________________________________________________

---

### CARD-T5: Action Game Pitch
- **Primary Genre:** Action (Secondary: Entertainment)
- **Client Download Size:** 111.82 MB
- **Network Architecture:** Single-Player / Asynchronous
- **Session Pacing:** Real-Time
- **Target Content Rating:** 4+ (Minimum OS: iOS 7.0)
- **Localization Scope:** 1 languages
- **Days Since Last Update:** 340 days
- **SHAP Plot Reference:** Panel 5 in `artifacts/figures/human_sim_treatment_cards.png`

**Your Prediction:** [ ] Class 0 (Ad-Supported)  |  [ ] Class 1 (Aggressive IAP)
**Confidence (1-5):** ____
**Key Rationale:** ____________________________________________________

---
