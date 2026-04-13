# Credit Card Fraud Detection
### Machine Learning · Explainability · Fairness Auditing · Interactive Dashboard

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-orange)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit)](https://streamlit.io/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-green)](https://shap.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Medium](https://img.shields.io/badge/Medium-Blog%20Post-black?logo=medium)](https://medium.com/p/30f7170dc11c)

---

## The Problem

Every two seconds, a credit card fraud attempt occurs somewhere in the world. Financial institutions lose over **$33 billion annually** to card fraud, yet aggressive fraud-blocking systems frustrate millions of legitimate customers who have their transactions unnecessarily declined.

This project tackles that tension head-on: **build a model that catches fraud without crying wolf**, then understand *why* it makes each decision, and verify it treats all customer segments fairly.

---

## Table of Contents

1. [Installations](#1-installations)
2. [Project Motivation](#2-project-motivation)
3. [CRISP-DM Process](#3-crisp-dm-process)
4. [File Descriptions](#4-file-descriptions)
5. [How to Interact with the Project](#5-how-to-interact-with-the-project)
6. [Key Results & Findings](#6-key-results--findings)
7. [Licensing, Authors & Acknowledgements](#7-licensing-authors--acknowledgements)

---

## 1. Installations

### Prerequisites

- Python **3.8 or higher**
- pip or conda package manager
- ~2 GB free RAM (for SHAP interaction values)

### Install Dependencies

```bash
pip install xgboost imbalanced-learn shap aequitas scikit-learn \
            pandas numpy matplotlib seaborn streamlit
```

Or using the provided requirements file:

```bash
pip install -r requirements.txt
```

### Core Library Versions Tested

| Library | Version | Purpose |
|---|---|---|
| `pandas` | 1.5+ | Data loading, manipulation, feature engineering |
| `numpy` | 1.23+ | Numerical operations, rolling statistics |
| `matplotlib` | 3.6+ | Static visualisations and chart exports |
| `seaborn` | 0.12+ | Statistical heatmaps (confusion matrix, parity) |
| `scikit-learn` | 1.2+ | Train/test split, metrics (F1, AUC, precision, recall) |
| `xgboost` | 1.7+ | Gradient-boosted classifier (primary model) |
| `imbalanced-learn` | 0.10+ | SMOTE oversampling, Random Undersampling |
| `shap` | 0.41+ | Model-agnostic feature attribution (TreeExplainer) |
| `aequitas` | 1.1.0 | Fairness & bias auditing framework |
| `streamlit` | 1.25+ | Interactive web dashboard |

### Dataset

Download `creditcard.csv` from [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the same directory as the notebook.

> **Note:** The dataset is ~144 MB and is not included in this repository due to size and licensing.

---

## 2. Project Motivation

### Why This Dataset?

The Kaggle Credit Card Fraud dataset is a canonical benchmark for real-world **class imbalance** problems — only **0.17%** of the 284,807 transactions are fraudulent. This extreme skew makes it the perfect stress-test for imbalance-handling strategies, and the business context (real financial cost of errors) makes it ideal for threshold optimisation and cost analysis.

### Why These Techniques?

Standard machine learning defaults (accuracy, 0.5 threshold) completely fail on imbalanced data — a model that predicts "legitimate" for every transaction achieves 99.83% accuracy while catching zero fraud. This project deliberately explores three fundamentally different approaches to the same problem, compares them honestly, and makes the comparison interactive.

### The Five Business Questions Driving This Analysis

> These questions were defined before any modelling began, following CRISP-DM business understanding principles.

**Q1 — Feature Importance**
> *What are the most important features, what do they mean, and how do they drive the predicted outcome?*

Understanding *what the model is actually learning* is essential for regulatory compliance and operational trust. SHAP values provide theoretically-grounded, consistent attribution for every individual prediction.

**Q2 — Creative Insights**
> *What unusual or creative insights can be gathered from the dataset?*

Beyond the standard model metrics, what patterns in the raw data reveal genuine fraud behaviour? Night-time timing, velocity bursting, and small-amount card testing are all explored.

**Q3 — Model Accuracy**
> *How accurate is the model that has been trained?*

A rigorous multi-metric evaluation (Precision, Recall, F1, AUC-ROC) across three techniques, with threshold sensitivity analysis and a business cost framework to identify the *operationally optimal* threshold — not just the statistically best one.

**Q4 — Predictive Scenario**
> *What will happen in a creative, predictive scenario using the trained model?*

A simulated card-testing attack demonstrates how the model's fraud probability escalates in real time as a fraudster increases transaction velocity during a late-night burst — showing the model working as an early-warning system.

**Q5 — Cost Optimisation**
> *What is the real dollar cost of model errors, and what decision threshold minimises total business loss?*

A missed fraud costs $500; a false alarm costs $5. That 100:1 asymmetry means the standard 0.50 threshold is too lenient. A threshold sweep identifies the cost-optimal operating point for each model variant.

---

## 3. CRISP-DM Process

This project follows the **Cross-Industry Standard Process for Data Mining (CRISP-DM)** framework throughout.

```
╔════════════════════════════════════════════════════════════════╗
║                    CRISP-DM CYCLE                              ║
║                                                                ║
║   ┌─────────────────────────────────────────────────────┐      ║
║   │  1. Business Understanding  →  Q1–Q4 defined        │      ║
║   │  2. Data Understanding      →  EDA, class balance   │      ║
║   │  3. Data Preparation        →  Feature engineering  │      ║
║   │                                 Missing values       │      ║
║   │                                 SMOTE / Undersamp.  │      ║
║   │  4. Modeling                →  3× XGBoost variants  │      ║
║   │  5. Evaluation              →  Metrics, SHAP,       │      ║
║   │                                 Cost, Fairness       │      ║
║   │  6. Deployment              →  Streamlit dashboard  │      ║
║   └─────────────────────────────────────────────────────┘      ║
╚════════════════════════════════════════════════════════════════╝
```

### Phase 1 · Business Understanding
Defined four concrete business questions (Q1–Q4 above). Established the asymmetric cost framework: a missed fraud costs **$500**; a false alarm costs **$5**. This ratio directly informs the choice of decision threshold.

### Phase 2 · Data Understanding
Explored the dataset structure (284,807 rows, 31 columns), confirmed zero missing values, and quantified the class imbalance (492 fraud / 284,315 legitimate). Identified the anonymised PCA features (V1–V28) and the two raw features (Amount, Time).

### Phase 3 · Data Preparation
Engineered nine new features from `Time` and `Amount`:

| Feature | Description | Rationale |
|---|---|---|
| `Hour` | Hour of day (0–23) | Time-of-day risk signal |
| `Hour_sin` / `Hour_cos` | Cyclical encoding of Hour | Prevents model treating Hour 23 as far from Hour 0 |
| `Is_Night` | 1 if hour < 6 or > 22 | High-risk window flag |
| `Time_Diff` | Seconds since previous transaction | Short gaps signal burst activity |
| `Time_Diff_Change` | Second-order time difference | Acceleration in transaction speed |
| `Tx_Count_1hr` | Transaction count in past 3,600 s | Velocity burst detector |
| `Amount_Mean_1hr` | Rolling 50-row mean of Amount | Contextual spend baseline |
| `Amount_Std_1hr` | Rolling 50-row std of Amount | Deviation from normal spend |

**Missing value strategy:** Rolling windows filled with 0 where insufficient history exists. This is appropriate: zero velocity genuinely means no prior transactions in the window — it is not missing data in the conventional sense.

**Resampling was applied only to the training split** to prevent data leakage.

### Phase 4 · Modeling
Three XGBoost classifiers trained with different imbalance-handling strategies:

| Method | Approach | Key Trade-off |
|---|---|---|
| `scale_pos_weight` | Native XGBoost weighting of minority class errors | Fastest; no new data; may under-represent fraud structure |
| SMOTE | Synthetic minority oversampling (k=5 neighbours) | Larger, more balanced training set; risk of over-fitting on synthetic points |
| Random Undersampling | Remove majority-class samples | Tiny training set; fast; loses legitimate transaction patterns |

All models evaluated at **threshold = 0.30** (deliberately lower than 0.50 to favour recall given the asymmetric cost structure).

### Phase 5 · Evaluation
- Precision, Recall, F1-Score, AUC-ROC for all three models
- Confusion matrices and ROC curves
- **Business cost analysis** with threshold sweep to find cost-optimal operating point
- **SHAP explainability** — global feature importance and pairwise interactions
- **Aequitas fairness audit** — disparity ratios and parity tests across synthetic protected attributes (spending tier, time-of-day segment, velocity quartile)
- **FairnessExperimentToolkit** — per-group threshold optimisation to reduce TPR disparity

### Phase 6 · Deployment
A five-tab interactive Streamlit dashboard allowing stakeholders to:
- Switch between the three model variants in real time
- Adjust the decision threshold and immediately see metric and cost changes
- Explore SHAP attributions and temporal patterns interactively

---

## 4. File Descriptions

```
fraud-detection/
│
├── fraud_detection_crisp_dm.ipynb      ← Main analysis notebook (run this first)
├── fraud_detection_fb_dashboard.html   ← Exported HTML view of the notebook
├── README.md                           ← This file
│
├── creditcard.csv                      ← Dataset (download from Kaggle — not included)
│
└── outputs/                            ← Generated on notebook run
    ├── class_distribution.png          ← Bar chart: fraud vs legitimate counts
    ├── technique_comparison.png        ← Grouped bars: Precision/Recall/F1/AUC
    ├── confusion_matrices.png          ← Side-by-side confusion matrices
    ├── shap_bar.png                    ← Global SHAP feature importance
    ├── shap_beeswarm.png               ← SHAP beeswarm (direction + magnitude)
    ├── shap_interaction.png            ← Top pairwise SHAP interaction
    ├── creative_insights.png           ← Temporal & amount anomaly charts (Q2)
    ├── velocity_insight.png            ← Tx_Count_1hr: fraud vs legitimate
    ├── predictive_scenario.png         ← Card-testing attack simulation (Q4)
    ├── parity_heatmap.png              ← Aequitas disparity heatmap
    └── threshold_sensitivity_*.png     ← Per-attribute fairness sensitivity curves
```

### Notebook Cell Map

| Cell Range | CRISP-DM Phase | Content |
|---|---|---|
| Cell 1 | Setup | Imports — all libraries |
| Cells 2–6 | Data Understanding | Load dataset, EDA, missing-value check, class balance |
| Cells 7–8 | Data Preparation | Feature engineering, train/test split |
| Cells 9–16 | Modeling | Three XGBoost variants + evaluation helper |
| Cells 17–20 | Evaluation | Comparison tables, bar charts, confusion matrices, cost analysis |
| Cells 21–22 | Evaluation | SHAP global importance, beeswarm, interactions |
| Cells 23–24 | Evaluation | Q2 creative insights (temporal + velocity patterns) |
| Cell 25 | Evaluation | Q4 predictive scenario (card-testing attack simulation) |
| Cells 26–34 | Evaluation (Fairness) | Aequitas group metrics, disparity, parity tests, heatmap |
| Cell 35 | Evaluation (Fairness) | FairnessExperimentToolkit class + per-attribute runs |
| Cell 36 | Evaluation (Fairness) | Consolidated audit summary |
| Cell 37 | Deployment | Full Streamlit dashboard source |

---

## 5. How to Interact with the Project

### Option A — Run the Jupyter Notebook (Recommended for Analysis)

```bash
# 1. Clone or download this repository
git clone https://github.com/eaamankwah/datascience-blog-post.git
cd datascience-blog-post

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place creditcard.csv in the project root

# 4. Launch Jupyter
jupyter notebook fraud_detection_crisp_dm.ipynb

# 5. Run all cells: Kernel → Restart & Run All
```

> **Runtime:** Expect approximately 5–10 minutes for the full notebook run (SMOTE, SHAP interaction values, and Aequitas fairness sweeps are the slowest steps).

### Option B — Interactive Streamlit Dashboard

The final notebook cell contains information about the full Streamlit app. To launch it independently:

```bash
# 1. Make sure that the Python script "app_explainability.py" is under your foot directory:

# 2. Run the dashboard
streamlit run app_explainability.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

**Dashboard tabs and what you can do:**

| Tab | Interactive Controls | What to Explore |
|---|---|---|
| 📊 Overview | Model selector, threshold slider | Watch Precision/Recall trade-off as you move the threshold |
| ⚖️ Technique Comparison | Threshold slider | See how all three models respond simultaneously |
| 💰 Cost Analysis | FP/FN cost inputs, model selector | Find the cost-optimal threshold for your business cost structure |
| 🔍 SHAP Explainability | Model selector | Compare which features each technique relies on most |
| ⏱️ Temporal Analysis | (Active model from sidebar) | Explore hourly patterns and velocity distributions |

---

## 6. Key Results & Findings

### Model Performance (threshold = 0.30)

| Technique | Precision | Recall | F1 | AUC-ROC | Est. Total Cost |
|---|---|---|---|---|---|
| scale_pos_weight | ~0.87 | ~0.82 | ~0.84 | ~0.978 | Lower FP cost |
| **SMOTE** | **~0.84** | **~0.87** | **~0.85** | **~0.979** | **Balanced** |
| Random Undersampling | ~0.22 | ~0.91 | ~0.35 | ~0.974 | Highest (FP-driven) |

> Exact values vary with your environment and dataset version. Run the notebook to reproduce.

### Q1 — Top Fraud Predictors (SHAP)
The strongest fraud signals are the **anonymised PCA features V4, V11, V12, V14, and V17**. Among interpretable features, **`Tx_Count_1hr`** (transaction velocity) and **`Amount`** are the most influential engineered predictors. High velocity in the past hour and anomalously small transaction amounts both significantly increase the fraud score.

### Q2 — Creative Insights Found
1. **Night-time fraud is ~2–3× more prevalent** than daytime fraud as a proportion of all transactions (midnight–6 AM is the highest-risk window).
2. **Fraud transactions cluster at very small amounts** (under $10) — consistent with card-testing behaviour where stolen cards are validated with micro-purchases before high-value fraud.
3. **Transaction velocity bursting** (high `Tx_Count_1hr`) is significantly more common in fraud transactions — fraudsters make rapid-fire purchases before a stolen card is blocked.

### Q3 — Model Accuracy Summary
All three XGBoost variants achieve **AUC-ROC > 0.97**, confirming strong discriminative power despite extreme class imbalance. SMOTE delivers the best F1 balance. The cost-optimal threshold is typically **0.18–0.25** (lower than the default 0.30), reflecting the high cost of missed fraud relative to false alarms.

### Q4 — Predictive Scenario
The card-testing attack simulation showed the model escalating fraud probability from ~0.05 to above the 0.30 threshold within **3–5 transactions**, as the attacker increases velocity and maintains late-night timing. This suggests real-time deployment could flag attacks **before** a high-value fraudulent purchase occurs.

### Fairness Audit Summary
Using three synthetic proxy attributes (spending tier, time-of-day segment, velocity quartile), the Aequitas audit found:
- Most groups pass TPR parity within the [0.80, 1.25] bounds at the global threshold.
- The **FairnessExperimentToolkit** identified per-group threshold adjustments that reduce inter-group TPR gaps by up to **0.08–0.12 points** for some segments, with minimal impact on overall model performance.

---

## 7. Licensing, Authors & Acknowledgements

### License
This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.  
You are free to use, modify, and distribute this code with attribution.

### Author
**Edward Amankwah**
Data Analytics Engineer | Gen AI BI Analyst | Data Scientist & ML Cloud Data Solutions

[GitHub](https://github.com/eaamankwah) 

[Medium](https://medium.com/@eaamankwah/how-your-bank-fights-fraud-behind-the-scenes-30f7170dc11c)

### Published Writing
- [How Your Bank Fights Fraud Behind the Scenes](https://medium.com/p/30f7170dc11c) — Medium, April 2026

### Dataset
> Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson and Gianluca Bontempi.  
> *Calibrating Probability with Undersampling for Unbalanced Classification.*  
> In Symposium on Computational Intelligence and Data Mining (CIDM), IEEE, 2015.

The dataset was collected and analysed by Worldline and the Machine Learning Group of ULB (Université Libre de Bruxelles).  
Available at: [https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

### Libraries & Tools
| Tool | Reference |
|---|---|
| XGBoost | Chen & Guestrin (2016), *XGBoost: A Scalable Tree Boosting System* |
| SHAP | Lundberg & Lee (2017), *A Unified Approach to Interpreting Model Predictions* |
| SMOTE | Chawla et al. (2002), *SMOTE: Synthetic Minority Over-sampling Technique* |
| Aequitas | Saleiro et al. (2018), *Aequitas: A Bias and Fairness Audit Toolkit* |
| Streamlit | [streamlit.io](https://streamlit.io/) |
| scikit-learn | Pedregosa et al. (2011), JMLR 12:2825–2830 |

### Acknowledgements
- **Udacity Data Science Nanodegree** — project structure and CRISP-DM framework guidance.
- The **MLG-ULB research group** for curating and sharing the dataset.
- The **open-source community** behind SHAP, Aequitas, and imbalanced-learn.

---

<div align="center">

*Built as part of an academic data science project — 2026*

</div>
