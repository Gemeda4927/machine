"""
PatientOutcome AI - Streamlit UI
Run:  pip install streamlit pandas numpy matplotlib scikit-learn
      streamlit run app.py

Uses synthetic educational data. Predictions are not medical advice.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------
# Config
# ----------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
TRAIN_FILE = BASE_DIR / "patient_outcomes_training.csv"
UNSEEN_FILE = BASE_DIR / "patient_outcomes_unseen.csv"

FEATURES = [
    "age",
    "glucose",
    "hemoglobin",
    "white_blood_cells",
    "heart_rate",
    "prior_admissions",
]
TARGET = "outcome"
RANDOM_STATE = 42

INK = "#12303A"
TEAL = "#0E7C86"
AMBER = "#C98A1B"
PAPER = "#F5F8F8"
MUTED = "#5B7078"
LINE = "#D5E0E2"

st.set_page_config(
    page_title="PatientOutcome AI",
    page_icon=None,
    layout="wide",
)

# ----------------------------------------------------------
# Styling
# ----------------------------------------------------------
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=Newsreader:wght@500;600&display=swap');

    html, body, [class*="css"], .stApp {{
        font-family: 'IBM Plex Sans', sans-serif;
        color: {INK};
    }}
    .stApp {{ background: {PAPER}; }}
    .block-container {{ padding-top: 2rem; max-width: 1150px; }}

    .hero {{
        background: {INK};
        color: #EAF3F4;
        border-radius: 14px;
        padding: 28px 34px;
        margin-bottom: 22px;
    }}
    .hero h1 {{
        font-family: 'Newsreader', serif;
        font-weight: 600;
        font-size: 2.3rem;
        margin: 0 0 6px 0;
        color: #FFFFFF;
    }}
    .hero p {{ margin: 0; color: #B9CFD3; font-size: 0.98rem; }}

    .stat {{
        background: #FFFFFF;
        border: 1px solid {LINE};
        border-left: 5px solid {TEAL};
        border-radius: 10px;
        padding: 14px 18px;
    }}
    .stat.warn {{ border-left-color: {AMBER}; }}
    .stat .label {{ color: {MUTED}; font-size: 0.82rem; }}
    .stat .value {{
        font-family: 'Newsreader', serif;
        font-size: 1.9rem;
        font-weight: 600;
        line-height: 1.2;
    }}

    .status {{
        display: inline-block;
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 500;
        margin-right: 8px;
        border: 1px solid {LINE};
        background: #FFFFFF;
        color: {MUTED};
    }}
    .status.ok {{ background: #E3F3F4; color: {TEAL}; border-color: #B7DEE1; }}

    .stButton > button, .stDownloadButton > button {{
        background: {TEAL};
        color: #FFFFFF;
        border: none;
        border-radius: 8px;
        padding: 0.65rem 1.6rem;
        font-weight: 600;
        font-size: 1rem;
    }}
    .stButton > button:hover, .stDownloadButton > button:hover {{
        background: {INK};
        color: #FFFFFF;
    }}
    .stButton > button:focus-visible {{ outline: 3px solid {AMBER}; }}

    .stTabs [data-baseweb="tab"] {{ font-weight: 500; }}
    .note {{ color: {MUTED}; font-size: 0.85rem; }}
    </style>
    """,
    unsafe_allow_html=True,
)


def stat(label, value, warn=False):
    cls = "stat warn" if warn else "stat"
    st.markdown(
        f'<div class="{cls}"><div class="label">{label}</div>'
        f'<div class="value">{value}</div></div>',
        unsafe_allow_html=True,
    )


def style_ax(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(LINE)
    ax.tick_params(colors=MUTED)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(INK)


# ----------------------------------------------------------
# Data helpers
# ----------------------------------------------------------
def make_sample_data(n_train=600, n_unseen=40):
    """Create synthetic training and unseen CSVs."""
    rng = np.random.default_rng(RANDOM_STATE)

    def build(n, labeled):
        df = pd.DataFrame(
            {
                "age": rng.integers(18, 90, n),
                "glucose": rng.normal(110, 30, n).round(1),
                "hemoglobin": rng.normal(13.5, 1.8, n).round(1),
                "white_blood_cells": rng.normal(7.5, 2.2, n).round(1),
                "heart_rate": rng.normal(80, 14, n).round(0),
                "prior_admissions": rng.poisson(1.2, n),
            }
        )
        score = (
            0.04 * (df["age"] - 55)
            + 0.02 * (df["glucose"] - 110)
            - 0.35 * (df["hemoglobin"] - 13.5)
            + 0.15 * (df["white_blood_cells"] - 7.5)
            + 0.02 * (df["heart_rate"] - 80)
            + 0.5 * (df["prior_admissions"] - 1)
            + rng.normal(0, 1.0, n)
        )
        if labeled:
            df[TARGET] = (score > 0.3).astype(int)
        mask = rng.random(df[FEATURES].shape) < 0.02
        df[FEATURES] = df[FEATURES].mask(mask)
        return df

    build(n_train, True).to_csv(TRAIN_FILE, index=False)
    build(n_unseen, False).to_csv(UNSEEN_FILE, index=False)


def read_source(upload, path):
    if upload is not None:
        return pd.read_csv(upload)
    if path.exists():
        return pd.read_csv(path)
    return None


def clean(df, with_target):
    """Validate columns, coerce to numeric, and check labels."""
    df = df.copy()
    required = FEATURES + ([TARGET] if with_target else [])
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("CSV is empty.")

    if with_target:
        df = df.drop_duplicates().copy()

    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[FEATURES] = df[FEATURES].replace([np.inf, -np.inf], np.nan)

    if df[FEATURES].isnull().all(axis=1).any():
        raise ValueError("Some rows have no usable feature values.")

    if with_target:
        df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
        if df[TARGET].isnull().any() or not df[TARGET].isin([0, 1]).all():
            raise ValueError("Outcome labels must be 0 or 1.")
        if df[TARGET].nunique() < 2:
            raise ValueError("Training data needs both outcome classes.")
        df[TARGET] = df[TARGET].astype(int)

    return df


def build_model():
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            ),
        ]
    )


def train_pipeline(train_df, test_size):
    X = train_df[FEATURES]
    y = train_df[TARGET]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )

    eval_model = build_model().fit(X_tr, y_tr)
    pred = eval_model.predict(X_te)
    proba = eval_model.predict_proba(X_te)[:, 1]

    final_model = build_model().fit(X, y)

    return {
        "model": final_model,
        "accuracy": accuracy_score(y_te, pred),
        "auc": roc_auc_score(y_te, proba) if y_te.nunique() == 2 else None,
        "matrix": confusion_matrix(y_te, pred, labels=[0, 1]),
        "report": pd.DataFrame(
            classification_report(
                y_te,
                pred,
                labels=[0, 1],
                target_names=["Category 0", "Category 1"],
                zero_division=0,
                output_dict=True,
            )
        )
        .T.round(3),
        "roc": roc_curve(y_te, proba) if y_te.nunique() == 2 else None,
        "n_train": len(X_tr),
        "n_test": len(X_te),
        "counts": y.value_counts().sort_index(),
        "coef": pd.Series(
            final_model.named_steps["classifier"].coef_[0], index=FEATURES
        ).sort_values(),
    }


# ----------------------------------------------------------
# Sidebar
# ----------------------------------------------------------
with st.sidebar:
    st.header("Data")
    train_upload = st.file_uploader("Training CSV (with outcome)", type="csv")
    unseen_upload = st.file_uploader("Unseen CSV (no outcome)", type="csv")
    st.caption(
        "Without uploads, the app reads the two CSV files placed beside app.py."
    )

    if st.button("Generate sample data", use_container_width=True):
        make_sample_data()
        st.success("Sample CSV files created.")

    st.header("Settings")
    test_size = st.slider("Evaluation split", 0.10, 0.40, 0.20, 0.05)

    st.caption("Synthetic educational data. Not medical advice.")

# ----------------------------------------------------------
# Load data
# ----------------------------------------------------------
train_raw = read_source(train_upload, TRAIN_FILE)
unseen_raw = read_source(unseen_upload, UNSEEN_FILE)

# ----------------------------------------------------------
# Header
# ----------------------------------------------------------
st.markdown(
    """
    <div class="hero">
      <h1>PatientOutcome AI</h1>
      <p>Train a classifier on labeled patient records, then score unseen patients.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

trained = "result" in st.session_state
st.markdown(
    f'<span class="status {"ok" if train_raw is not None else ""}">'
    f'Training data {"ready" if train_raw is not None else "missing"}</span>'
    f'<span class="status {"ok" if unseen_raw is not None else ""}">'
    f'Unseen data {"ready" if unseen_raw is not None else "missing"}</span>'
    f'<span class="status {"ok" if trained else ""}">'
    f'Model {"trained" if trained else "not trained"}</span>',
    unsafe_allow_html=True,
)
st.write("")

tab_data, tab_train, tab_predict = st.tabs(
    ["Data", "Train and evaluate", "Predict unseen"]
)

# ----------------------------------------------------------
# Tab 1: Data
# ----------------------------------------------------------
with tab_data:
    if train_raw is None:
        st.info(
            "No training data found. Upload a CSV in the sidebar or select "
            "Generate sample data."
        )
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            stat("Training rows", f"{len(train_raw):,}")
        with c2:
            stat("Missing values", f"{int(train_raw.isnull().sum().sum()):,}")
        with c3:
            stat("Duplicate rows", f"{int(train_raw.duplicated().sum()):,}")

        st.subheader("Training preview")
        st.dataframe(train_raw.head(20), use_container_width=True)

        if TARGET in train_raw.columns:
            counts = (
                pd.to_numeric(train_raw[TARGET], errors="coerce")
                .value_counts()
                .sort_index()
            )
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.bar(
                [f"Category {int(i)}" for i in counts.index],
                counts.values,
                color=[TEAL, AMBER][: len(counts)],
            )
            ax.set_ylabel("Records")
            ax.set_title("Outcome distribution")
            style_ax(ax)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    if unseen_raw is not None:
        st.subheader("Unseen preview")
        st.dataframe(unseen_raw.head(10), use_container_width=True)

# ----------------------------------------------------------
# Tab 2: Train
# ----------------------------------------------------------
with tab_train:
    if st.button("Train model", type="primary", key="train"):
        if train_raw is None:
            st.error("Training data is missing.")
        else:
            try:
                with st.spinner("Training"):
                    clean_train = clean(train_raw, with_target=True)
                    st.session_state["result"] = train_pipeline(
                        clean_train, test_size
                    )
                    st.session_state.pop("predictions", None)
                st.success("Model trained on all labeled records.")
            except ValueError as err:
                st.error(str(err))

    res = st.session_state.get("result")
    if res is None:
        st.markdown(
            '<p class="note">Train the model to see accuracy, ROC-AUC, '
            "and the confusion matrix.</p>",
            unsafe_allow_html=True,
        )
    else:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            stat("Accuracy", f"{res['accuracy']:.1%}")
        with c2:
            stat("ROC-AUC", f"{res['auc']:.3f}" if res["auc"] else "n/a")
        with c3:
            stat("Fit samples", f"{res['n_train']:,}")
        with c4:
            stat("Evaluation samples", f"{res['n_test']:,}", warn=True)

        st.write("")
        left, right = st.columns(2)

        with left:
            fig, ax = plt.subplots(figsize=(4.6, 4))
            ax.imshow(res["matrix"], cmap="GnBu")
            for (i, j), v in np.ndenumerate(res["matrix"]):
                ax.text(
                    j,
                    i,
                    int(v),
                    ha="center",
                    va="center",
                    fontsize=16,
                    color=INK,
                )
            ax.set_xticks([0, 1], ["Category 0", "Category 1"])
            ax.set_yticks([0, 1], ["Category 0", "Category 1"])
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title("Confusion matrix")
            style_ax(ax)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with right:
            if res["roc"] is not None:
                fpr, tpr, _ = res["roc"]
                fig, ax = plt.subplots(figsize=(4.6, 4))
                ax.plot(fpr, tpr, color=TEAL, linewidth=2.5)
                ax.plot([0, 1], [0, 1], color=LINE, linestyle="--")
                ax.set_xlabel("False positive rate")
                ax.set_ylabel("True positive rate")
                ax.set_title("ROC curve")
                style_ax(ax)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

        left, right = st.columns(2)
        with left:
            st.subheader("Classification report")
            st.dataframe(res["report"], use_container_width=True)
        with right:
            st.subheader("Feature influence")
            fig, ax = plt.subplots(figsize=(4.6, 3))
            ax.barh(
                res["coef"].index,
                res["coef"].values,
                color=[AMBER if v > 0 else TEAL for v in res["coef"].values],
            )
            ax.axvline(0, color=LINE)
            ax.set_xlabel("Coefficient (scaled features)")
            style_ax(ax)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.caption("Amber raises Category 1 probability; teal lowers it.")

# ----------------------------------------------------------
# Tab 3: Predict
# ----------------------------------------------------------
with tab_predict:
    if st.button("Test on unseen data", type="primary", key="predict"):
        if "result" not in st.session_state:
            st.error("Train the model first.")
        elif unseen_raw is None:
            st.error("Unseen data is missing.")
        else:
            try:
                clean_unseen = clean(unseen_raw, with_target=False)
                model = st.session_state["result"]["model"]
                X_new = clean_unseen[FEATURES]

                out = clean_unseen.copy()
                out["predicted_outcome"] = model.predict(X_new)
                out["probability_category_1"] = (
                    model.predict_proba(X_new)[:, 1].round(4)
                )
                out["predicted_label"] = out["predicted_outcome"].map(
                    {0: "Category 0", 1: "Category 1"}
                )
                st.session_state["predictions"] = out
            except ValueError as err:
                st.error(str(err))

    preds = st.session_state.get("predictions")
    if preds is None:
        st.markdown(
            '<p class="note">Select Test on unseen data to score the unseen '
            "records with the trained model.</p>",
            unsafe_allow_html=True,
        )
    else:
        n1 = int((preds["predicted_outcome"] == 1).sum())
        n0 = len(preds) - n1
        c1, c2, c3 = st.columns(3)
        with c1:
            stat("Records scored", f"{len(preds):,}")
        with c2:
            stat("Category 0", f"{n0:,}")
        with c3:
            stat("Category 1", f"{n1:,}", warn=True)

        st.write("")
        fig, ax = plt.subplots(figsize=(7, 2.8))
        ax.hist(
            preds["probability_category_1"],
            bins=20,
            range=(0, 1),
            color=TEAL,
            edgecolor="white",
        )
        ax.axvline(0.5, color=AMBER, linestyle="--", linewidth=2)
        ax.set_xlabel("Probability of Category 1")
        ax.set_ylabel("Patients")
        ax.set_title("Prediction confidence")
        style_ax(ax)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Predictions")
        st.dataframe(
            preds.style.background_gradient(
                subset=["probability_category_1"], cmap="YlOrBr", vmin=0, vmax=1
            ),
            use_container_width=True,
        )

        c1, c2 = st.columns([1, 3])
        with c1:
            st.download_button(
                "Download CSV",
                preds.to_csv(index=False).encode("utf-8"),
                file_name="patient_outcome_predictions.csv",
                mime="text/csv",
            )
        with c2:
            if st.button("Save beside app.py"):
                preds.to_csv(BASE_DIR / "patient_outcome_predictions.csv", index=False)
                st.success("Saved patient_outcome_predictions.csv")