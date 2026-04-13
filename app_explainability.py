# ============================================================
# app_explainability.py
# Credit Card Fraud Detection — Enhanced Streamlit Dashboard
# Covers: scale_pos_weight | SMOTE | Random Undersampling
# Run: streamlit run app_explainability.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
TECH_COLORS = {
    "scale_pos_weight": "#2E5F8A",
    "SMOTE":            "#4CAF50",
    "Undersampling":    "#E67E22"
}
COST_FP_DEFAULT = 5
COST_FN_DEFAULT = 500

# ─────────────────────────────────────────────
# DATA LOADING & FEATURE ENGINEERING (cached)
# ─────────────────────────────────────────────
@st.cache_data
def load_and_engineer():
    df = pd.read_csv("creditcard.csv")
    df["Hour"]             = (df["Time"] // 3600) % 24
    df["Is_Night"]         = df["Hour"].apply(lambda x: 1 if x < 6 or x > 22 else 0)
    df                     = df.sort_values("Time")
    df["Time_Diff"]        = df["Time"].diff().fillna(0)
    df["Time_Diff_Change"] = df["Time_Diff"].diff().fillna(0)
    df["Hour_sin"]         = np.sin(2 * np.pi * df["Hour"] / 24)
    df["Hour_cos"]         = np.cos(2 * np.pi * df["Hour"] / 24)
    window = 50
    df["Tx_Count_1hr"]    = df["Time"].rolling(window=3600).count().fillna(0)
    df["Amount_Mean_1hr"] = df["Amount"].rolling(window=window).mean().fillna(0)
    df["Amount_Std_1hr"]  = df["Amount"].rolling(window=window).std().fillna(0)
    return df

@st.cache_data
def get_split(df):
    v_cols   = [c for c in df.columns if c.startswith("V")]
    eng_cols = ["Amount","Hour","Is_Night","Time_Diff","Time_Diff_Change",
                "Tx_Count_1hr","Amount_Mean_1hr","Amount_Std_1hr","Hour_sin","Hour_cos"]
    features = eng_cols + v_cols
    X, y = df[features], df["Class"]
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# ─────────────────────────────────────────────
# MODEL TRAINING (each technique cached separately)
# ─────────────────────────────────────────────
@st.cache_resource
def train_spw(X_train, y_train):
    spw = len(y_train[y_train==0]) / len(y_train[y_train==1])
    m = XGBClassifier(scale_pos_weight=spw, eval_metric="logloss",
                      use_label_encoder=False, random_state=42)
    m.fit(X_train, y_train)
    return m

@st.cache_resource
def train_smote(X_train, y_train):
    Xs, ys = SMOTE(k_neighbors=5, random_state=42).fit_resample(X_train, y_train)
    m = XGBClassifier(eval_metric="logloss", use_label_encoder=False, random_state=42)
    m.fit(Xs, ys)
    return m, int(sum(ys==0)), int(sum(ys==1))

@st.cache_resource
def train_rus(X_train, y_train):
    Xr, yr = RandomUnderSampler(random_state=42).fit_resample(X_train, y_train)
    m = XGBClassifier(eval_metric="logloss", use_label_encoder=False, random_state=42)
    m.fit(Xr, yr)
    return m, int(sum(yr==0)), int(sum(yr==1))

# ─────────────────────────────────────────────
# EVALUATION HELPER
# ─────────────────────────────────────────────
def evaluate(model, X_test, y_test, threshold, cost_fp, cost_fn):
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    fpr_arr, tpr_arr, _ = roc_curve(y_test, probs)
    return {
        "probs":     probs,
        "preds":     preds,
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall":    recall_score(y_test, preds, zero_division=0),
        "f1":        f1_score(y_test, preds, zero_division=0),
        "auc":       auc(fpr_arr, tpr_arr),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "cost":      fp * cost_fp + fn * cost_fn,
        "fpr":       fpr_arr,
        "tpr":       tpr_arr,
    }

# ─────────────────────────────────────────────
# LOAD DATA & MODELS
# ─────────────────────────────────────────────
df = load_and_engineer()
X_train, X_test, y_train, y_test = get_split(df)

with st.spinner("Training models — this runs once then caches..."):
    model_spw                        = train_spw(X_train, y_train)
    model_smote, smote_leg, smote_fr = train_smote(X_train, y_train)
    model_rus,   rus_leg,   rus_fr   = train_rus(X_train, y_train)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
st.sidebar.title("⚙️ Controls")

st.sidebar.subheader("Active model")
active_model_name = st.sidebar.radio(
    "Select technique to inspect",
    ["scale_pos_weight", "SMOTE", "Undersampling"],
    index=0
)
model_map = {
    "scale_pos_weight": model_spw,
    "SMOTE":            model_smote,
    "Undersampling":    model_rus
}
active_model = model_map[active_model_name]

st.sidebar.subheader("Decision threshold")
threshold = st.sidebar.slider("Threshold", 0.01, 0.99, 0.30, 0.01,
    help="Lower = catch more fraud (more false positives). Higher = fewer false positives (more missed fraud).")

st.sidebar.subheader("Business costs")
cost_fp = st.sidebar.number_input("Cost per False Positive ($)", value=COST_FP_DEFAULT, min_value=1)
cost_fn = st.sidebar.number_input("Cost per False Negative ($)", value=COST_FN_DEFAULT, min_value=1)

st.sidebar.subheader("Threshold sweep")
sweep_model_name = st.sidebar.selectbox(
    "Sweep thresholds for",
    ["scale_pos_weight", "SMOTE", "Undersampling"]
)

# ─────────────────────────────────────────────
# EVALUATE ALL THREE AT THE CHOSEN THRESHOLD
# ─────────────────────────────────────────────
results = {
    "scale_pos_weight": evaluate(model_spw,   X_test, y_test, threshold, cost_fp, cost_fn),
    "SMOTE":            evaluate(model_smote, X_test, y_test, threshold, cost_fp, cost_fn),
    "Undersampling":    evaluate(model_rus,   X_test, y_test, threshold, cost_fp, cost_fn),
}
active = results[active_model_name]

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.title("💳 Credit Card Fraud Detection Dashboard")
st.caption(f"Threshold: **{threshold:.2f}**  |  FP cost: **${cost_fp}**  |  FN cost: **${cost_fn}**  |  Active model: **{active_model_name}**")
st.divider()

# ══════════════════════════════════════════════
# TAB LAYOUT
# ══════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview",
    "⚖️ Technique Comparison",
    "💰 Cost Analysis",
    "🔍 SHAP Explainability",
    "⏱️ Temporal Analysis"
])

# ─────────────────────────────────────────────
# TAB 1 — OVERVIEW (active model)
# ─────────────────────────────────────────────
with tab1:
    st.subheader(f"Active model: {active_model_name}")

    # KPI metric cards
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Precision",  f"{active['precision']:.3f}")
    c2.metric("Recall",     f"{active['recall']:.3f}")
    c3.metric("F1-Score",   f"{active['f1']:.3f}")
    c4.metric("AUC-ROC",    f"{active['auc']:.4f}")
    c5.metric("Total Cost", f"${active['cost']:,}")

    st.divider()
    col_cm, col_roc = st.columns(2)

    # Confusion matrix heatmap
    with col_cm:
        st.markdown("**Confusion matrix**")
        cm = np.array([[active["tn"], active["fp"]],
                       [active["fn"], active["tp"]]])
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Pred Legit", "Pred Fraud"],
                    yticklabels=["Actual Legit", "Actual Fraud"])
        ax.set_title(active_model_name, fontsize=10)
        st.pyplot(fig)
        plt.close()
        st.caption(f"TP={active['tp']}  FP={active['fp']}  FN={active['fn']}  TN={active['tn']}")

    # ROC curve for all three models
    with col_roc:
        st.markdown("**ROC curves — all three techniques**")
        fig2, ax2 = plt.subplots(figsize=(4, 3))
        for name, res in results.items():
            ax2.plot(res["fpr"], res["tpr"],
                     color=TECH_COLORS[name], linewidth=2,
                     label=f"{name} (AUC={res['auc']:.4f})")
        ax2.plot([0,1],[0,1], "k--", linewidth=0.8)
        ax2.set_xlabel("False Positive Rate")
        ax2.set_ylabel("True Positive Rate")
        ax2.legend(fontsize=7)
        ax2.set_title("ROC Curves", fontsize=10)
        st.pyplot(fig2)
        plt.close()

    # Threshold sensitivity for active model
    st.divider()
    st.markdown("**Precision / Recall vs threshold (active model)**")
    thresholds_range = np.linspace(0.01, 0.99, 80)
    prec_list, rec_list, f1_list = [], [], []
    probs_active = active_model.predict_proba(X_test)[:, 1]
    for t in thresholds_range:
        p_t = (probs_active >= t).astype(int)
        prec_list.append(precision_score(y_test, p_t, zero_division=0))
        rec_list.append(recall_score(y_test, p_t, zero_division=0))
        f1_list.append(f1_score(y_test, p_t, zero_division=0))

    fig3, ax3 = plt.subplots(figsize=(8, 3))
    ax3.plot(thresholds_range, prec_list, label="Precision", color="#2E5F8A")
    ax3.plot(thresholds_range, rec_list,  label="Recall",    color="#4CAF50")
    ax3.plot(thresholds_range, f1_list,   label="F1-Score",  color="#E67E22")
    ax3.axvline(threshold, color="red", linestyle="--", linewidth=1.2,
                label=f"Current threshold ({threshold:.2f})")
    ax3.set_xlabel("Threshold")
    ax3.set_ylabel("Score")
    ax3.legend(fontsize=8)
    ax3.set_title(f"Threshold sensitivity — {active_model_name}", fontsize=10)
    st.pyplot(fig3)
    plt.close()

# ─────────────────────────────────────────────
# TAB 2 — TECHNIQUE COMPARISON
# ─────────────────────────────────────────────
with tab2:
    st.subheader("Side-by-side technique comparison")
    st.caption(f"All evaluated at threshold = {threshold:.2f}")

    # Summary metrics table
    rows = []
    for name, res in results.items():
        rows.append({
            "Technique":   name,
            "Precision":   f"{res['precision']:.3f}",
            "Recall":      f"{res['recall']:.3f}",
            "F1-Score":    f"{res['f1']:.3f}",
            "AUC-ROC":     f"{res['auc']:.4f}",
            "TP": res["tp"], "FP": res["fp"],
            "FN": res["fn"], "TN": res["tn"],
            "Total Cost":  f"${res['cost']:,}"
        })
    st.dataframe(pd.DataFrame(rows).set_index("Technique"), use_container_width=True)

    st.divider()
    col_bar, col_cm3 = st.columns([1.2, 1.8])

    # Grouped bar chart — Precision / Recall / F1 / AUC
    with col_bar:
        st.markdown("**Performance metrics**")
        metrics_labels = ["Precision", "Recall", "F1-Score", "AUC-ROC"]
        x = np.arange(len(metrics_labels))
        width = 0.25
        fig4, ax4 = plt.subplots(figsize=(5, 4))
        for i, (name, res) in enumerate(results.items()):
            vals = [res["precision"], res["recall"], res["f1"], res["auc"]]
            ax4.bar(x + i*width, vals, width, label=name,
                    color=TECH_COLORS[name], edgecolor="white")
        ax4.set_xticks(x + width)
        ax4.set_xticklabels(metrics_labels, fontsize=9)
        ax4.set_ylim(0, 1.05)
        ax4.legend(fontsize=7)
        ax4.set_title("Metrics by technique", fontsize=10)
        st.pyplot(fig4)
        plt.close()

    # Three confusion matrices side by side
    with col_cm3:
        st.markdown("**Confusion matrices**")
        fig5, axes = plt.subplots(1, 3, figsize=(8, 2.8))
        for ax, (name, res) in zip(axes, results.items()):
            cm = np.array([[res["tn"], res["fp"]],
                           [res["fn"], res["tp"]]])
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                        xticklabels=["Leg","Fra"],
                        yticklabels=["Leg","Fra"],
                        cbar=False)
            ax.set_title(name, fontsize=8)
        plt.tight_layout()
        st.pyplot(fig5)
        plt.close()

    # Training set sizes bar chart
    st.divider()
    st.markdown("**Training set composition after resampling**")
    train_data = {
        "scale_pos_weight": (int(sum(y_train==0)), int(sum(y_train==1))),
        "SMOTE":            (smote_leg, smote_fr),
        "Undersampling":    (rus_leg,   rus_fr),
    }
    fig6, ax6 = plt.subplots(figsize=(8, 3))
    names  = list(train_data.keys())
    leg_v  = [train_data[n][0] for n in names]
    fra_v  = [train_data[n][1] for n in names]
    x6     = np.arange(len(names))
    ax6.bar(x6, leg_v, label="Legitimate", color="#B5D4F4")
    ax6.bar(x6, fra_v, bottom=leg_v, label="Fraud", color="#E24B4A")
    ax6.set_xticks(x6)
    ax6.set_xticklabels(names)
    ax6.set_ylabel("Sample count")
    ax6.legend()
    for xi, (l, f) in zip(x6, zip(leg_v, fra_v)):
        ax6.text(xi, l+f+1000, f"Total: {l+f:,}", ha="center", fontsize=8)
    ax6.set_title("Training set size by technique", fontsize=10)
    st.pyplot(fig6)
    plt.close()

    # FP vs FN trade-off scatter
    st.divider()
    st.markdown("**False positive vs false negative trade-off**")
    fig7, ax7 = plt.subplots(figsize=(6, 3.5))
    for name, res in results.items():
        ax7.scatter(res["fp"], res["fn"], s=200,
                    color=TECH_COLORS[name], zorder=5, label=name)
        ax7.annotate(name, (res["fp"], res["fn"]),
                     textcoords="offset points", xytext=(8, 4), fontsize=8)
    ax7.set_xlabel("False Positives (blocked legit transactions)")
    ax7.set_ylabel("False Negatives (missed fraud)")
    ax7.set_title("FP vs FN trade-off", fontsize=10)
    ax7.legend(fontsize=8)
    st.pyplot(fig7)
    plt.close()
    st.caption("Bottom-left is ideal (fewest errors of both types). Note: Undersampling trades extreme FP increase for FN reduction.")

# ─────────────────────────────────────────────
# TAB 3 — COST ANALYSIS
# ─────────────────────────────────────────────
with tab3:
    st.subheader("Business cost analysis")
    st.caption(f"FP cost = ${cost_fp}  |  FN cost = ${cost_fn}  |  Threshold = {threshold:.2f}")

    # Cost breakdown cards
    c1, c2, c3 = st.columns(3)
    for col, (name, res) in zip([c1, c2, c3], results.items()):
        fp_cost = res["fp"] * cost_fp
        fn_cost = res["fn"] * cost_fn
        col.metric(name, f"${res['cost']:,}",
                   delta=f"FP ${fp_cost:,} + FN ${fn_cost:,}",
                   delta_color="inverse")

    st.divider()

    # Stacked cost bar chart
    st.markdown("**Cost breakdown by error type**")
    fig8, ax8 = plt.subplots(figsize=(7, 4))
    names8  = list(results.keys())
    fp_costs = [results[n]["fp"] * cost_fp for n in names8]
    fn_costs = [results[n]["fn"] * cost_fn for n in names8]
    x8 = np.arange(len(names8))
    ax8.bar(x8, fp_costs, label=f"FP cost (${cost_fp} each)", color="#EF9F27")
    ax8.bar(x8, fn_costs, bottom=fp_costs, label=f"FN cost (${cost_fn} each)", color="#E24B4A")
    ax8.set_xticks(x8)
    ax8.set_xticklabels(names8)
    ax8.set_ylabel("Total cost ($)")
    ax8.legend()
    for xi, (fpc, fnc) in zip(x8, zip(fp_costs, fn_costs)):
        ax8.text(xi, fpc+fnc+100, f"${fpc+fnc:,}", ha="center", fontsize=9, fontweight="bold")
    ax8.set_title("Business cost breakdown by technique", fontsize=10)
    st.pyplot(fig8)
    plt.close()

    # Threshold sweep cost chart for selected model
    st.divider()
    st.markdown(f"**Cost vs threshold sweep — {sweep_model_name}**")
    sweep_model = model_map[sweep_model_name]
    sweep_probs = sweep_model.predict_proba(X_test)[:, 1]
    sweep_thresholds = np.linspace(0.01, 0.5, 40)
    sweep_costs, sweep_fp_counts, sweep_fn_counts = [], [], []
    for t in sweep_thresholds:
        p_t = (sweep_probs >= t).astype(int)
        tn_t, fp_t, fn_t, tp_t = confusion_matrix(y_test, p_t).ravel()
        sweep_costs.append(fp_t * cost_fp + fn_t * cost_fn)
        sweep_fp_counts.append(fp_t)
        sweep_fn_counts.append(fn_t)

    min_idx  = int(np.argmin(sweep_costs))
    opt_thresh = sweep_thresholds[min_idx]
    opt_cost   = sweep_costs[min_idx]

    fig9, ax9 = plt.subplots(figsize=(8, 3.5))
    ax9.plot(sweep_thresholds, sweep_costs,
             color=TECH_COLORS[sweep_model_name], linewidth=2, label="Total cost")
    ax9.axvline(opt_thresh, color="green", linestyle="--",
                label=f"Optimal: {opt_thresh:.3f} (${opt_cost:,})")
    ax9.axvline(threshold, color="red", linestyle=":",
                label=f"Current: {threshold:.2f}")
    ax9.set_xlabel("Threshold")
    ax9.set_ylabel("Total cost ($)")
    ax9.legend(fontsize=8)
    ax9.set_title(f"Cost curve — {sweep_model_name}", fontsize=10)
    st.pyplot(fig9)
    plt.close()

    st.info(f"**Optimal threshold for {sweep_model_name}:** {opt_thresh:.3f} → Total cost ${opt_cost:,}  "
            f"(FP={sweep_fp_counts[min_idx]}, FN={sweep_fn_counts[min_idx]})")

    # Cross-model optimal threshold table
    st.divider()
    st.markdown("**Optimal threshold comparison across all techniques**")
    opt_rows = []
    for name, model in model_map.items():
        prbs = model.predict_proba(X_test)[:, 1]
        costs_t, fp_t_list, fn_t_list = [], [], []
        for t in sweep_thresholds:
            pt = (prbs >= t).astype(int)
            tn_t, fp_t, fn_t, _ = confusion_matrix(y_test, pt).ravel()
            costs_t.append(fp_t*cost_fp + fn_t*cost_fn)
            fp_t_list.append(fp_t)
            fn_t_list.append(fn_t)
        mi = int(np.argmin(costs_t))
        opt_rows.append({
            "Technique":         name,
            "Optimal Threshold": f"{sweep_thresholds[mi]:.3f}",
            "FP at Optimal":     fp_t_list[mi],
            "FN at Optimal":     fn_t_list[mi],
            "Min Cost":          f"${costs_t[mi]:,}"
        })
    st.dataframe(pd.DataFrame(opt_rows).set_index("Technique"), use_container_width=True)

# ─────────────────────────────────────────────
# TAB 4 — SHAP EXPLAINABILITY
# ─────────────────────────────────────────────
with tab4:
    st.subheader("SHAP explainability")
    st.caption("SHAP computed on a 500-sample random subset of the test set.")

    shap_model_name = st.radio(
        "Select model for SHAP analysis",
        ["scale_pos_weight", "SMOTE", "Undersampling"],
        horizontal=True
    )
    shap_model = model_map[shap_model_name]

    with st.spinner("Computing SHAP values..."):
        X_sample = X_test.sample(500, random_state=42)
        explainer = shap.TreeExplainer(shap_model)
        shap_vals = explainer.shap_values(X_sample)

    col_s1, col_s2 = st.columns(2)

    with col_s1:
        st.markdown("**Global feature importance (bar)**")
        fig_s1, ax_s1 = plt.subplots(figsize=(5, 5))
        shap.summary_plot(shap_vals, X_sample, plot_type="bar",
                          show=False, max_display=15)
        st.pyplot(fig_s1)
        plt.close()

    with col_s2:
        st.markdown("**SHAP beeswarm (impact direction)**")
        fig_s2, ax_s2 = plt.subplots(figsize=(5, 5))
        shap.summary_plot(shap_vals, X_sample, show=False, max_display=15)
        st.pyplot(fig_s2)
        plt.close()

    # SHAP feature importance comparison across techniques
    st.divider()
    st.markdown("**Top-10 SHAP feature importance comparison across techniques**")
    importance_data = {}
    for name, model in model_map.items():
        sv = shap.TreeExplainer(model).shap_values(X_sample)
        mean_abs = np.abs(sv).mean(axis=0)
        importance_data[name] = pd.Series(mean_abs, index=X_sample.columns)

    df_imp = pd.DataFrame(importance_data)
    top10  = df_imp.mean(axis=1).nlargest(10).index

    fig_imp, ax_imp = plt.subplots(figsize=(8, 4))
    x_imp = np.arange(len(top10))
    w_imp = 0.25
    for i, (name, ser) in enumerate(df_imp.loc[top10].items()):
        ax_imp.bar(x_imp + i*w_imp, ser.values, w_imp,
                   label=name, color=TECH_COLORS[name])
    ax_imp.set_xticks(x_imp + w_imp)
    ax_imp.set_xticklabels(top10, rotation=30, ha="right", fontsize=8)
    ax_imp.set_ylabel("Mean |SHAP value|")
    ax_imp.legend(fontsize=8)
    ax_imp.set_title("Top-10 feature importance by technique", fontsize=10)
    st.pyplot(fig_imp)
    plt.close()
    st.caption("Differences in SHAP importance across techniques reveal how each resampling strategy shifts the model's reliance on different features.")

    # Interactions
    st.divider()
    st.markdown("**Top SHAP feature interaction**")
    with st.spinner("Computing interaction values..."):
        shap_interact = explainer.shap_interaction_values(X_sample)
    interact_strength = np.abs(shap_interact).mean(axis=0)
    np.fill_diagonal(interact_strength, 0)
    i_idx, j_idx = np.unravel_index(np.argmax(interact_strength), interact_strength.shape)
    feat_i = X_sample.columns[i_idx]
    feat_j = X_sample.columns[j_idx]
    st.write(f"Top interacting features: **{feat_i}** & **{feat_j}**")
    fig_int, ax_int = plt.subplots(figsize=(6, 4))
    shap.dependence_plot((i_idx, j_idx), shap_interact, X_sample,
                         feature_names=X_sample.columns, show=False, ax=ax_int)
    st.pyplot(fig_int)
    plt.close()

# ─────────────────────────────────────────────
# TAB 5 — TEMPORAL ANALYSIS
# ─────────────────────────────────────────────
with tab5:
    st.subheader("Temporal patterns in transaction data")

    fraud     = df[df["Class"] == 1]
    non_fraud = df[df["Class"] == 0]

    col_t1, col_t2 = st.columns(2)

    with col_t1:
        st.markdown("**Transaction volume by hour**")
        fig_t1, ax_t1 = plt.subplots(figsize=(5, 3))
        ax_t1.hist(non_fraud["Hour"], bins=24, alpha=0.6, label="Legitimate", color="#2E5F8A")
        ax_t1.hist(fraud["Hour"],     bins=24, alpha=0.8, label="Fraud",      color="#E24B4A")
        ax_t1.set_xlabel("Hour of Day")
        ax_t1.set_ylabel("Count")
        ax_t1.legend()
        ax_t1.set_title("Fraud vs Legitimate by Hour", fontsize=10)
        st.pyplot(fig_t1)
        plt.close()

    with col_t2:
        st.markdown("**Night-time fraud rate**")
        night_fraud   = len(fraud[fraud["Is_Night"] == 1])
        night_legit   = len(non_fraud[non_fraud["Is_Night"] == 1])
        day_fraud     = len(fraud[fraud["Is_Night"] == 0])
        day_legit     = len(non_fraud[non_fraud["Is_Night"] == 0])
        night_rate    = night_fraud / (night_fraud + night_legit) * 100
        day_rate      = day_fraud   / (day_fraud   + day_legit)   * 100
        fig_t2, ax_t2 = plt.subplots(figsize=(4, 3))
        ax_t2.bar(["Daytime", "Night-time"], [day_rate, night_rate],
                  color=["#2E5F8A", "#E24B4A"])
        ax_t2.set_ylabel("Fraud rate (%)")
        ax_t2.set_title("Fraud rate: Day vs Night", fontsize=10)
        for xi, val in enumerate([day_rate, night_rate]):
            ax_t2.text(xi, val+0.002, f"{val:.3f}%", ha="center", fontsize=9)
        st.pyplot(fig_t2)
        plt.close()

    st.divider()
    col_t3, col_t4 = st.columns(2)

    with col_t3:
        st.markdown("**Transaction amount distribution (fraud vs legit)**")
        fig_t3, ax_t3 = plt.subplots(figsize=(5, 3))
        ax_t3.hist(non_fraud["Amount"].clip(0, 500), bins=50,
                   alpha=0.6, label="Legitimate", color="#2E5F8A", density=True)
        ax_t3.hist(fraud["Amount"].clip(0, 500),     bins=50,
                   alpha=0.8, label="Fraud",      color="#E24B4A", density=True)
        ax_t3.set_xlabel("Amount ($, clipped at $500)")
        ax_t3.set_ylabel("Density")
        ax_t3.legend()
        ax_t3.set_title("Amount distribution", fontsize=10)
        st.pyplot(fig_t3)
        plt.close()

    with col_t4:
        st.markdown("**Fraud detection rate by hour (active model)**")
        probs_all = active_model.predict_proba(X_test)[:, 1]
        preds_all = (probs_all >= threshold).astype(int)
        X_test_copy          = X_test.copy()
        X_test_copy["actual"]    = y_test.values
        X_test_copy["predicted"] = preds_all
        hourly = X_test_copy.groupby("Hour").apply(
            lambda g: g["predicted"][g["actual"]==1].mean() if g["actual"].sum() > 0 else 0
        ).reset_index()
        hourly.columns = ["Hour", "DetectionRate"]
        fig_t4, ax_t4 = plt.subplots(figsize=(5, 3))
        ax_t4.bar(hourly["Hour"], hourly["DetectionRate"],
                  color=TECH_COLORS[active_model_name])
        ax_t4.set_xlabel("Hour of Day")
        ax_t4.set_ylabel("Fraud detection rate")
        ax_t4.set_title(f"Detection rate by hour — {active_model_name}", fontsize=10)
        st.pyplot(fig_t4)
        plt.close()

    # Rolling transaction velocity
    st.divider()
    st.markdown("**Rolling transaction velocity (sample of test set)**")
    sample_idx = X_test.sample(500, random_state=1).index
    fig_t5, ax_t5 = plt.subplots(figsize=(9, 2.5))
    ax_t5.plot(range(500), X_test.loc[sample_idx, "Tx_Count_1hr"].values,
               color="#2E5F8A", linewidth=0.8, alpha=0.8)
    ax_t5.set_xlabel("Transaction index (sample)")
    ax_t5.set_ylabel("Tx count / hr")
    ax_t5.set_title("Transaction velocity (Tx_Count_1hr) — 500-sample window", fontsize=10)
    st.pyplot(fig_t5)
    plt.close()

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.divider()
st.caption("Credit Card Fraud Detection Dashboard  |  XGBoost + SMOTE + Random Undersampling + SHAP  |  Academic Project 2026")
