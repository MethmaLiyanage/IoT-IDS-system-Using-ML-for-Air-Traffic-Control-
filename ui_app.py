import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os
import sys
import joblib

sys.path.append("src")

from train_random_forest import train_random_forest
from train_decision_tree import train_decision_tree
from data_loader import load_and_preprocess_data
from config import ROLLING_WINDOW
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

RESULTS_DIR = os.path.join("src", "results")

st.set_page_config(
    page_title="IoT Intrusion Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CLASS_NAMES  = ["Normal", "Backdoor", "Password", "DDoS", "Injection", "Ransomware", "XSS", "Scanning"]
CLASS_COLORS = ["#16a34a", "#f59e0b", "#9333ea", "#dc2626", "#2563eb", "#ec4899", "#0d9488", "#ca8a04"]
SHORT_NAMES  = ["Norm", "Back", "Pass", "DDoS", "Inj", "Rans", "XSS", "Scan"]
LABEL_MAP    = {i: n for i, n in enumerate(CLASS_NAMES)}
REVERSE_MAP  = {n: i for i, n in enumerate(CLASS_NAMES)}
COLOR_MAP    = {n: c for n, c in zip(CLASS_NAMES, CLASS_COLORS)}

SENSOR_COLS = ["temperature", "pressure", "humidity", "latitude", "longitude"]
ROLL_COLS   = [f"{c}_{s}" for c in SENSOR_COLS for s in ["roll_mean", "roll_std"]]


@st.cache_resource(show_spinner="Loading data and models...")
def load_and_train():
    X, y, feature_names = load_and_preprocess_data(
        "data/Final_Fused_IoT_Dataset_FIXED.csv"
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    rf_path = os.path.join(RESULTS_DIR, "rf_model.pkl")
    dt_path = os.path.join(RESULTS_DIR, "dt_model.pkl")

    if os.path.exists(rf_path) and os.path.exists(dt_path):
        rf_model = joblib.load(rf_path)
        dt_model = joblib.load(dt_path)
    else:
        rf_model = train_random_forest(X_train, y_train)
        dt_model = train_decision_tree(X_train, y_train)
        joblib.dump(rf_model, rf_path)
        joblib.dump(dt_model, dt_path)

    return X, y, feature_names, X_train, X_test, y_train, y_test, rf_model, dt_model


X, y, feature_names, X_train, X_test, y_train, y_test, rf_model, dt_model = load_and_train()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("IoT IDS")
    st.caption("Intrusion Detection System")
    st.divider()
    page = st.radio(
        "Navigation",
        ["IDS Overview", "Manual Input", "Upload & Predict", "Model Analysis"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("**Dataset**")
    st.caption(f"Total samples: {len(y):,}")
    st.caption(f"Features: {len(feature_names)}")
    st.caption(f"Train: {len(y_train):,} | Test: {len(y_test):,}")

# ─── DASHBOARD ────────────────────────────────────────────────────────────────
if page == "IDS Overview":
    st.title("IDS Overview")
    st.caption("Overview of the IoT dataset and attack distribution")

    unique, counts = np.unique(y, return_counts=True)
    count_dict = dict(zip(unique, counts))
    total = len(y)

    # Metrics row
    cols = st.columns(4)
    cols[0].metric("Total Samples", f"{total:,}")
    cols[1].metric("Normal", f"{count_dict.get(0, 0):,}", f"{count_dict.get(0,0)/total*100:.1f}%")
    cols[2].metric("Attack Types", "7")
    cols[3].metric("Total Attacks", f"{total - count_dict.get(0, 0):,}",
                   f"{(total - count_dict.get(0,0))/total*100:.1f}%")

    st.divider()
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Attack Distribution")
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor("#0e1117")
        ax.set_facecolor("#0e1117")
        labels = [CLASS_NAMES[i] for i in unique]
        colors = [CLASS_COLORS[i] for i in unique]
        total = sum(counts)
        legend_labels = [f"{l}  {c/total*100:.1f}%" for l, c in zip(labels, counts)]
        wedges, _ = ax.pie(
            counts, labels=None, colors=colors,
            startangle=90, wedgeprops={"edgecolor": "#0e1117", "linewidth": 2},
        )
        ax.legend(wedges, legend_labels, loc="center left", bbox_to_anchor=(1, 0.5),
                  fontsize=8, facecolor="#0e1117", labelcolor="white", framealpha=0.3)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col_right:
        st.subheader("Attack Count by Type")
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        fig2.patch.set_facecolor("#0e1117")
        ax2.set_facecolor("#161b27")
        bar_labels = [CLASS_NAMES[i] for i in unique]
        bar_colors = [CLASS_COLORS[i] for i in unique]
        bars = ax2.bar(bar_labels, counts, color=bar_colors, edgecolor="#0e1117")
        ax2.set_ylabel("Count", color="white")
        ax2.tick_params(colors="white", labelsize=7)
        ax2.set_xticklabels(bar_labels, rotation=30, ha="right", fontsize=7)
        for spine in ax2.spines.values():
            spine.set_edgecolor("#374151")
        for bar, cnt in zip(bars, counts):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
                     f"{cnt:,}", ha="center", va="bottom", color="white", fontsize=7)
        fig2.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close()

    st.divider()
    st.subheader("Sensor Feature Correlation")
    df_sample = pd.DataFrame(X[:1000], columns=feature_names)
    base_cols = [c for c in SENSOR_COLS if c in df_sample.columns]
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    fig3.patch.set_facecolor("#0e1117")
    ax3.set_facecolor("#0e1117")
    corr = df_sample[base_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax3,
                linewidths=0.5, cbar_kws={"shrink": 0.8})
    ax3.tick_params(colors="white")
    for label in ax3.get_xticklabels() + ax3.get_yticklabels():
        label.set_color("white")
    st.pyplot(fig3, use_container_width=True)
    plt.close()

    st.divider()
    st.subheader("Sensor Distributions by Attack Type")
    df_vis = pd.DataFrame(X, columns=feature_names)
    df_vis["label"] = y
    plot_features = [f for f in SENSOR_COLS if f in df_vis.columns]
    fig4, axes = plt.subplots(1, len(plot_features), figsize=(14, 3))
    fig4.patch.set_facecolor("#0e1117")
    for i, feat in enumerate(plot_features):
        ax = axes[i]
        ax.set_facecolor("#161b27")
        for cls_idx in unique:
            vals = df_vis.loc[df_vis["label"] == cls_idx, feat]
            ax.hist(vals, bins=30, alpha=0.5, color=CLASS_COLORS[cls_idx],
                    label=CLASS_NAMES[cls_idx], density=True)
        ax.set_title(feat.capitalize(), color="white", fontsize=9)
        ax.tick_params(colors="white", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#374151")
    handles = [mpatches.Patch(color=CLASS_COLORS[i], label=CLASS_NAMES[i]) for i in unique]
    fig4.legend(handles=handles, loc="lower center", ncol=4, fontsize=7,
                facecolor="#0e1117", labelcolor="white", framealpha=0.5,
                bbox_to_anchor=(0.5, -0.2))
    fig4.tight_layout()
    st.pyplot(fig4, use_container_width=True)
    plt.close()

# ─── MANUAL INPUT ─────────────────────────────────────────────────────────────
elif page == "Manual Input":
    import datetime
    st.title("Manual Sensor Input")
    st.caption(
        f"Enter {ROLLING_WINDOW} consecutive sensor readings — roll mean & std are computed automatically "
        f"(window = {ROLLING_WINDOW}, matching training). Prediction is made on Reading {ROLLING_WINDOW}."
    )

    # Initialise session state for prediction history
    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []

    col_model, _ = st.columns([1, 2])
    with col_model:
        model_choice = st.selectbox("Select Model", ["Random Forest", "Decision Tree"])

    # ── Load Example ──────────────────────────────────────────────────────────
    EXAMPLES = {
        "Normal": [
            (0.0, 0.0, 0.0, 0.0, 10.0),
            (31.79, 1.035, 32.04, 0.0, 0.0),
            (31.79, 1.035, 32.04, 0.0, 0.0),
            (0.0, 0.0, 0.0, 0.0, 10.0),
            (0.0, 0.0, 0.0, 0.0, 10.0),
        ],
        "Backdoor": [
            (0.0, 0.0, 0.0, 32.9168, 42.8959),
            (0.0, 0.0, 0.0, 50.6464, 61.5169),
            (27.14, 0.4279, 31.48, 0.0, 0.0),
            (0.0, 0.0, 0.0, 41.3474, 54.6755),
            (38.185, 1.035, 99.45, 0.0, 0.0),
        ],
        "Password": [
            (0.0, 0.0, 0.0, 549.38, 555.13),
            (0.0, 0.0, 0.0, 530.12, 552.26),
            (28.85, 19.95, 23.86, 0.0, 0.0),
            (35.44, -10.35, 40.49, 0.0, 0.0),
            (46.62, 1.035, 62.95, 0.0, 0.0),
        ],
        "DDoS": [
            (0.0, 0.0, 0.0, 8.0137, 18.7),
            (28.57, -8.14, 55.52, 0.0, 0.0),
            (31.50, 0.24, 78.90, 0.0, 0.0),
            (38.11, -0.93, 48.91, 0.0, 0.0),
            (36.36, 1.035, 97.99, 0.0, 0.0),
        ],
        "Injection": [
            (44.02, -1.33, 56.70, 0.0, 0.0),
            (33.41, 0.60, 91.81, 0.0, 0.0),
            (0.0, 0.0, 0.0, 10.41, 20.57),
            (29.81, 6.49, 28.49, 0.0, 0.0),
            (40.92, -1.41, 63.91, 0.0, 0.0),
        ],
        "Ransomware": [
            (37.13, 4.17, 27.33, 0.0, 0.0),
            (0.0, 0.0, 0.0, 64.54, 75.53),
            (37.18, 4.17, 27.33, 0.0, 0.0),
            (0.0, 0.0, 0.0, 64.59, 75.64),
            (45.77, 1.035, 4.51, 0.0, 0.0),
        ],
        "XSS": [
            (0.0, 0.0, 0.0, 367.39, 369.47),
            (0.0, 0.0, 0.0, 358.62, 373.13),
            (28.11, 18.00, 62.84, 0.0, 0.0),
            (45.35, 2.71, 84.51, 0.0, 0.0),
            (20.71, 1.035, 78.10, 0.0, 0.0),
        ],
        "Scanning": [
            (26.63, -3.24, 67.51, 0.0, 0.0),
            (0.0, 0.0, 0.0, 128.48, 138.10),
            (26.95, -3.24, 67.51, 0.0, 0.0),
            (0.0, 0.0, 0.0, 128.56, 138.11),
            (0.0, 0.0, 0.0, 128.63, 138.22),
        ],
    }

    st.divider()
    ex_col, _ = st.columns([1, 2])
    with ex_col:
        example_choice = st.selectbox(
            "Load Example (optional)",
            ["— select to pre-fill —"] + list(EXAMPLES.keys()),
            key="example_choice",
        )

    # Determine default values: from example or blank
    if example_choice != "— select to pre-fill —":
        ex_rows = EXAMPLES[example_choice]
    else:
        ex_rows = [(25.0, 1.035, 65.0, 0.0, 0.0)] * ROLLING_WINDOW

    st.subheader(f"{ROLLING_WINDOW} Consecutive Sensor Readings")
    st.caption(f"Reading {ROLLING_WINDOW} is the current/latest reading that will be classified.")

    COL_KEYS    = ["temperature", "pressure", "humidity", "latitude", "longitude"]
    COL_LABELS  = ["Temperature (°C)", "Pressure (hPa)", "Humidity (%)", "Latitude", "Longitude"]
    reading_vals = {k: [] for k in COL_KEYS}

    for i in range(1, ROLLING_WINDOW + 1):
        st.markdown(f"**Reading {i}{'  ← predict this' if i == ROLLING_WINDOW else ''}**")
        cols = st.columns(5)
        for j, (label, key) in enumerate(zip(COL_LABELS, COL_KEYS)):
            val = cols[j].number_input(
                label,
                value=float(ex_rows[i - 1][j]),
                step=0.1 if key not in ("latitude", "longitude") else 0.001,
                format="%.4f",
                key=f"{key}_r{i}_{example_choice}",
            )
            reading_vals[key].append(val)

    st.divider()

    if st.button("Predict Attack Type", type="primary"):
        model = rf_model if model_choice == "Random Forest" else dt_model

        roll_mean = {k: float(np.mean(reading_vals[k])) for k in COL_KEYS}
        roll_std  = {k: float(np.std(reading_vals[k], ddof=1)) if len(set(reading_vals[k])) > 1 else 0.0
                     for k in COL_KEYS}
        latest = {k: reading_vals[k][-1] for k in COL_KEYS}

        row = {}
        for col in feature_names:
            if col in COL_KEYS:
                row[col] = latest[col]
            elif "_roll_mean" in col:
                row[col] = roll_mean.get(col.replace("_roll_mean", ""), 0.0)
            elif "_roll_std" in col:
                row[col] = roll_std.get(col.replace("_roll_std", ""), 0.0)
            else:
                row[col] = 0.0

        X_input = np.array([[row[c] for c in feature_names]])
        pred = model.predict(X_input)[0]
        attack_name = LABEL_MAP[pred]
        color = COLOR_MAP[attack_name]
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Save to prediction history
        st.session_state.prediction_history.append({
            "Time":        ts,
            "Model":       model_choice,
            "Temp":        round(latest["temperature"], 2),
            "Pressure":    round(latest["pressure"], 3),
            "Humidity":    round(latest["humidity"], 2),
            "Latitude":    round(latest["latitude"], 4),
            "Longitude":   round(latest["longitude"], 4),
            "Prediction":  attack_name,
        })

        # Show computed rolling stats
        with st.expander("Computed Rolling Features (used by model)"):
            roll_df = pd.DataFrame({
                "Sensor":    COL_KEYS,
                "roll_mean": [round(roll_mean[k], 4) for k in COL_KEYS],
                "roll_std":  [round(roll_std[k],  4) for k in COL_KEYS],
            })
            st.dataframe(roll_df, use_container_width=True)

        st.divider()
        st.subheader("Detection Result")
        st.caption(f"Predicted on Reading {ROLLING_WINDOW} — {ts}")

        if attack_name == "Normal":
            st.success(f"No Attack Detected — **{attack_name}**")
        else:
            st.error(f"Attack Detected — **{attack_name}**")

        st.markdown(
            f"<div style='background-color:{color};padding:20px;border-radius:10px;"
            f"text-align:center;'>"
            f"<h2 style='color:white;margin:0;'>{attack_name}</h2>"
            f"</div>",
            unsafe_allow_html=True,
        )

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_input)[0]
            classes = model.classes_
            st.divider()
            st.subheader("Confidence Scores")
            proba_df = pd.DataFrame({
                "Attack Type": [CLASS_NAMES[c] for c in classes],
                "Confidence (%)": (proba * 100).round(2),
            }).sort_values("Confidence (%)", ascending=False)

            fig_conf, ax_conf = plt.subplots(figsize=(7, 3))
            fig_conf.patch.set_facecolor("#0e1117")
            ax_conf.set_facecolor("#161b27")
            bar_colors = [COLOR_MAP[n] for n in proba_df["Attack Type"]]
            ax_conf.barh(proba_df["Attack Type"], proba_df["Confidence (%)"],
                         color=bar_colors)
            ax_conf.set_xlabel("Confidence (%)", color="white")
            ax_conf.tick_params(colors="white")
            ax_conf.set_xlim(0, 105)
            for spine in ax_conf.spines.values():
                spine.set_edgecolor("#374151")
            for i, (val, name) in enumerate(zip(proba_df["Confidence (%)"], proba_df["Attack Type"])):
                if val > 0.5:
                    ax_conf.text(val + 0.5, i, f"{val:.1f}%", va="center",
                                 color="white", fontsize=8)
            fig_conf.tight_layout()
            st.pyplot(fig_conf, use_container_width=True)
            plt.close()

    # ── Prediction History ────────────────────────────────────────────────────
    if st.session_state.prediction_history:
        st.divider()
        st.subheader("Prediction History")
        hist_df = pd.DataFrame(st.session_state.prediction_history)

        def color_pred_hist(val):
            c = COLOR_MAP.get(val, "#374151")
            return f"background-color: {c}; color: white"

        styled_hist = hist_df.style.map(color_pred_hist, subset=["Prediction"])
        st.dataframe(styled_hist, use_container_width=True)

        csv_hist = hist_df.to_csv(index=False).encode("utf-8")
        col_dl, col_clear = st.columns([1, 1])
        with col_dl:
            st.download_button("Download History as CSV", data=csv_hist,
                               file_name="prediction_history.csv", mime="text/csv")
        with col_clear:
            if st.button("Clear History"):
                st.session_state.prediction_history = []
                st.rerun()

# ─── UPLOAD & PREDICT ─────────────────────────────────────────────────────────
elif page == "Upload & Predict":
    st.title("Upload & Predict")
    st.caption("Run attack detection on the pre-loaded dataset or upload a new CSV file")

    def color_pred(val):
        color = COLOR_MAP.get(val, "#374151")
        return f"background-color: {color}; color: white"

    def render_results(preds, y_true, X_input, model, model_choice, show_proba, display_df):
        pred_labels = [LABEL_MAP[p] for p in preds]
        display_df = display_df.copy()
        display_df = display_df.drop(columns=["attack_label", "type", "final_label", "device"], errors="ignore")
        display_df["Prediction"] = pred_labels
        if y_true is not None:
            display_df["Actual"] = [LABEL_MAP[int(v)] for v in y_true]
        if show_proba and hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_input)
            display_df["Confidence"] = (proba.max(axis=1) * 100).round(1).astype(str) + "%"

        st.success(f"Detection complete using **{model_choice}**")

        # Attack breakdown metrics
        vc = pd.Series(pred_labels).value_counts()
        row1 = st.columns(4)
        row1[0].metric("Normal",     f"{vc.get('Normal', 0):,}")
        row1[1].metric("Backdoor",   f"{vc.get('Backdoor', 0):,}")
        row1[2].metric("Password",   f"{vc.get('Password', 0):,}")
        row1[3].metric("DDoS",       f"{vc.get('DDoS', 0):,}")
        row2 = st.columns(4)
        row2[0].metric("Injection",  f"{vc.get('Injection', 0):,}")
        row2[1].metric("Ransomware", f"{vc.get('Ransomware', 0):,}")
        row2[2].metric("XSS",        f"{vc.get('XSS', 0):,}")
        row2[3].metric("Scanning",   f"{vc.get('Scanning', 0):,}")

        if y_true is not None:
            acc = accuracy_score(y_true, preds)
            st.metric("Accuracy vs True Labels", f"{acc*100:.2f}%")
            with st.expander("Classification Report"):
                report = classification_report(
                    y_true, preds, labels=list(range(8)),
                    target_names=CLASS_NAMES, output_dict=True
                )
                st.dataframe(pd.DataFrame(report).transpose().round(3), use_container_width=True)

        # Pie chart
        st.subheader("Prediction Breakdown")
        fig_p, ax_p = plt.subplots(figsize=(5, 4))
        fig_p.patch.set_facecolor("#0e1117")
        ax_p.set_facecolor("#0e1117")
        pie_colors = [COLOR_MAP[l] for l in vc.index]
        total_p = sum(vc.values)
        legend_labels_p = [f"{l}  {c/total_p*100:.1f}%" for l, c in zip(vc.index, vc.values)]
        wedges_p, _ = ax_p.pie(
            vc.values, labels=None,
            colors=pie_colors,
            wedgeprops={"edgecolor": "#0e1117", "linewidth": 2},
        )
        ax_p.legend(wedges_p, legend_labels_p, loc="center left", bbox_to_anchor=(1, 0.5),
                    fontsize=8, facecolor="#0e1117", labelcolor="white", framealpha=0.3)
        st.pyplot(fig_p, use_container_width=True)
        plt.close()

        st.subheader(f"Detection Results (showing first 500 of {len(display_df):,} rows)")
        preview = display_df.head(500)
        styled = preview.style.map(color_pred, subset=["Prediction"])
        if y_true is not None:
            styled = styled.map(color_pred, subset=["Actual"])
        st.dataframe(styled, use_container_width=True)

        csv_out = display_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Results as CSV",
            data=csv_out,
            file_name="detection_results.csv",
            mime="text/csv",
        )

    col_settings, col_mode = st.columns([1, 2])
    with col_settings:
        model_choice = st.selectbox("Select Model", ["Random Forest", "Decision Tree"])
    with col_mode:
        data_source = st.radio(
            "Data source",
            ["Use pre-loaded dataset (test split)", "Upload a new CSV file"],
            horizontal=True,
        )

    st.divider()

    if data_source == "Use pre-loaded dataset (test split)":
        st.info(
            f"Using the **test split** of the fused IoT dataset "
            f"({len(y_test):,} samples — stratified 20% split). "
            "True labels are available so accuracy will be shown."
        )

        if st.button("Run Detection on Pre-loaded Data", type="primary"):
            model = rf_model if model_choice == "Random Forest" else dt_model
            X_run = X_test
            y_run = y_test
            preds = model.predict(X_run)
            display_df = pd.DataFrame(X_run, columns=feature_names)
            render_results(preds, y_run, X_run, model, model_choice, True, display_df)

    else:
        uploaded_file = st.file_uploader(
            "Upload CSV file", type=["csv"],
            help="File should contain sensor columns: temperature, pressure, humidity, latitude, longitude",
        )

        if uploaded_file:
            df_upload = pd.read_csv(uploaded_file)
            st.subheader("Preview of Uploaded Data")
            st.dataframe(df_upload.head(10), use_container_width=True)

            required_cols = list(feature_names)
            has_true_labels = "attack_label" in df_upload.columns

            if "date" in df_upload.columns and "time" in df_upload.columns:
                st.info("Raw sensor file detected — applying preprocessing...")
                dfw = df_upload.copy()
                dfw["timestamp"] = pd.to_datetime(
                    dfw["date"].astype(str) + " " + dfw["time"].astype(str), format="mixed"
                )
                dfw = dfw.sort_values("timestamp").drop(
                    columns=["date", "time", "timestamp", "type", "final_label", "device"], errors="ignore"
                )
                window = ROLLING_WINDOW
                for col in SENSOR_COLS:
                    if col in dfw.columns:
                        dfw[f"{col}_roll_mean"] = dfw[col].rolling(window).mean()
                        dfw[f"{col}_roll_std"]  = dfw[col].rolling(window).std()
                dfw = dfw.fillna(0)
                y_upload = dfw["attack_label"].values if has_true_labels else None
                if has_true_labels:
                    dfw = dfw.drop(columns=["attack_label"])
                X_new = dfw[required_cols].values if all(c in dfw.columns for c in required_cols) else dfw.values
            else:
                y_upload = df_upload["attack_label"].values if has_true_labels else None
                cols_present = [c for c in required_cols if c in df_upload.columns]
                X_new = df_upload[cols_present].values if cols_present else df_upload.values

            # Negative pressure warning
            if "pressure" in df_upload.columns:
                neg_pressure_count = int((df_upload["pressure"] < 0).sum())
                if neg_pressure_count > 0:
                    st.warning(
                        f"⚠️ **{neg_pressure_count:,} rows have negative pressure values.** "
                        "Real sensors never produce negative pressure — these are synthetic attack signatures "
                        "(Injection and Scanning attacks in this dataset use negative pressure as a pattern). "
                        "The model will still predict correctly as it was trained on this data."
                    )

            if st.button("Run Detection", type="primary"):
                model = rf_model if model_choice == "Random Forest" else dt_model
                try:
                    preds = model.predict(X_new)
                    render_results(preds, y_upload, X_new, model, model_choice, True, df_upload)
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    st.info("Make sure your CSV has the same sensor columns as the training data.")
        else:
            st.info("Upload a CSV file to get started.")

# ─── MODEL ANALYSIS ───────────────────────────────────────────────────────────
elif page == "Model Analysis":
    st.title("Model Analysis")
    st.caption("Performance metrics, confusion matrices, and feature importance")

    model_tab, feat_tab, lstm_tab = st.tabs(
        ["Model Performance", "Feature Importance", "LSTM Results"]
    )

    with model_tab:
        rf_col, dt_col = st.columns(2)
        for col, model, name in [
            (rf_col, rf_model, "Random Forest"),
            (dt_col, dt_model, "Decision Tree"),
        ]:
            with col:
                st.subheader(name)
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                st.metric("Accuracy", f"{acc*100:.2f}%")
                report = classification_report(
                    y_test, y_pred, labels=list(range(8)),
                    target_names=CLASS_NAMES, output_dict=True
                )
                report_df = pd.DataFrame(report).transpose().round(3)
                st.dataframe(report_df, use_container_width=True)
                st.markdown("**Confusion Matrix**")
                cm = confusion_matrix(y_test, y_pred, labels=list(range(8)))
                cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
                fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
                fig_cm.patch.set_facecolor("#0e1117")
                ax_cm.set_facecolor("#0e1117")
                sns.heatmap(
                    cm_norm, annot=True, fmt=".2f", cmap="Blues", ax=ax_cm,
                    xticklabels=SHORT_NAMES, yticklabels=SHORT_NAMES,
                    linewidths=0.5, vmin=0, vmax=1,
                )
                ax_cm.set_xlabel("Predicted", color="white")
                ax_cm.set_ylabel("Actual", color="white")
                ax_cm.tick_params(colors="white", labelsize=7)
                for label in ax_cm.get_xticklabels() + ax_cm.get_yticklabels():
                    label.set_color("white")
                st.pyplot(fig_cm, use_container_width=True)
                plt.close()

        st.divider()
        st.subheader("Model Comparison")
        rf_pred = rf_model.predict(X_test)
        dt_pred = dt_model.predict(X_test)
        metrics = {}
        for mname, mpreds in [("Random Forest", rf_pred), ("Decision Tree", dt_pred)]:
            report = classification_report(
                y_test, mpreds, labels=list(range(8)),
                target_names=CLASS_NAMES, output_dict=True
            )
            metrics[mname] = {
                "Accuracy":    accuracy_score(y_test, mpreds),
                "Macro F1":    report["macro avg"]["f1-score"],
                "Weighted F1": report["weighted avg"]["f1-score"],
            }
        compare_df = pd.DataFrame(metrics).T * 100
        fig_bar, ax_bar = plt.subplots(figsize=(6, 3))
        fig_bar.patch.set_facecolor("#0e1117")
        ax_bar.set_facecolor("#161b27")
        x = np.arange(len(compare_df.columns))
        width = 0.35
        bars1 = ax_bar.bar(x - width/2, compare_df.loc["Random Forest"], width,
                           label="Random Forest", color="#3b82f6")
        bars2 = ax_bar.bar(x + width/2, compare_df.loc["Decision Tree"],  width,
                           label="Decision Tree",  color="#f59e0b")
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(compare_df.columns, color="white")
        ax_bar.set_ylabel("Score (%)", color="white")
        ax_bar.set_ylim(70, 102)
        ax_bar.tick_params(colors="white")
        ax_bar.legend(facecolor="#0e1117", labelcolor="white")
        for spine in ax_bar.spines.values():
            spine.set_edgecolor("#374151")
        for bar in list(bars1) + list(bars2):
            ax_bar.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.2,
                f"{bar.get_height():.1f}%",
                ha="center", va="bottom", color="white", fontsize=7,
            )
        st.pyplot(fig_bar, use_container_width=True)
        plt.close()

    with feat_tab:
        st.subheader("Top 10 Feature Importances — Random Forest")
        importances = rf_model.feature_importances_
        imp_df = pd.DataFrame({
            "Feature":    feature_names,
            "Importance": importances,
        }).sort_values("Importance", ascending=False).head(10)
        fig_fi, ax_fi = plt.subplots(figsize=(7, 5))
        fig_fi.patch.set_facecolor("#0e1117")
        ax_fi.set_facecolor("#161b27")
        ax_fi.barh(imp_df["Feature"][::-1], imp_df["Importance"][::-1], color="#3b82f6")
        ax_fi.set_xlabel("Importance Score", color="white")
        ax_fi.tick_params(colors="white")
        for spine in ax_fi.spines.values():
            spine.set_edgecolor("#374151")
        fig_fi.tight_layout()
        st.pyplot(fig_fi, use_container_width=True)
        plt.close()

        st.divider()
        st.subheader("Full Feature Importance Table")
        full_imp_df = pd.DataFrame({
            "Feature":       feature_names,
            "RF Importance": rf_model.feature_importances_.round(4),
            "DT Importance": dt_model.feature_importances_.round(4),
        }).sort_values("RF Importance", ascending=False).reset_index(drop=True)
        st.dataframe(full_imp_df, use_container_width=True)

    with lstm_tab:
        st.subheader("LSTM Model Results")
        st.caption("Results from the trained LSTM model saved during main.py execution")

        lstm_report_path = os.path.join(RESULTS_DIR, "LSTM_report.csv")
        lstm_matrix_path = os.path.join(RESULTS_DIR, "LSTM_matrix.png")

        if os.path.exists(lstm_report_path):
            lstm_df = pd.read_csv(lstm_report_path, index_col=0).round(3)
            acc_row = lstm_df.loc["accuracy"] if "accuracy" in lstm_df.index else None
            if acc_row is not None:
                st.metric("LSTM Accuracy", f"{float(acc_row['f1-score'])*100:.2f}%")
            st.markdown("**Classification Report**")
            st.dataframe(lstm_df, use_container_width=True)
        else:
            st.warning("LSTM report not found. Run `main.py` from the `src/` directory first.")

        if os.path.exists(lstm_matrix_path):
            st.markdown("**Confusion Matrix**")
            st.image(lstm_matrix_path, use_container_width=True)
        else:
            st.info("LSTM confusion matrix image not found.")
