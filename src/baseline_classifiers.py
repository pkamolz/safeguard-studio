"""
Baseline classifiers for multi-label toxicity detection.

Models:
  - TF-IDF (100K) + Logistic Regression (OvR)  → tfidf_lr.joblib
  - TF-IDF (20K)  + Logistic Regression (OvR)  → tfidf_lr_20k.joblib
  - TF-IDF (20K)  + XGBoost (OvR)              → tfidf_xgb.joblib
Data:   data/processed/train_cleaned.parquet
Output: models/, docs/figures/, W&B project 'safeguard-studio'
"""

import os
import warnings
from pathlib import Path

import dotenv
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
dotenv.load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_PATH = Path("data/processed/train_cleaned.parquet")
MODELS_DIR = Path("models")
FIGURES_DIR = Path("docs/figures")
WANDB_PROJECT = "safeguard-studio"
LABEL_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
RANDOM_STATE = 42
TEST_SIZE = 0.2

TFIDF_PARAMS = dict(
    max_features=20_000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=3,
    strip_accents="unicode",
    analyzer="word",
)

# LR can handle 100K features without OOM; XGBoost histogram buffers cannot
TFIDF_PARAMS_LR_100K = {**TFIDF_PARAMS, "max_features": 100_000}

LR_PARAMS = dict(
    C=1.0,
    max_iter=1000,
    solver="lbfgs",
    class_weight="balanced",
    random_state=RANDOM_STATE,
)

XGB_PARAMS = dict(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=RANDOM_STATE,
    n_jobs=2,       # capped — histogram buffers scale with n_features × n_threads; >2 OOMs at 20k features
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, labels: list[str]) -> dict:
    metrics: dict = {}
    for i, label in enumerate(labels):
        yt, yp, ypr = y_true[:, i], y_pred[:, i], y_prob[:, i]
        metrics[f"{label}/precision"] = precision_score(yt, yp, zero_division=0)
        metrics[f"{label}/recall"] = recall_score(yt, yp, zero_division=0)
        metrics[f"{label}/f1"] = f1_score(yt, yp, zero_division=0)
        try:
            metrics[f"{label}/auc_roc"] = roc_auc_score(yt, ypr)
        except ValueError:
            metrics[f"{label}/auc_roc"] = float("nan")

    metrics["macro_f1"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["weighted_f1"] = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    try:
        metrics["macro_auc_roc"] = roc_auc_score(y_true, y_prob, average="macro", multi_class="ovr")
    except ValueError:
        metrics["macro_auc_roc"] = float("nan")
    return metrics


def save_confusion_matrices(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str], model_name: str) -> list[Path]:
    saved: list[Path] = []
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"Confusion Matrices — {model_name}", fontsize=14, fontweight="bold")
    for i, (label, ax) in enumerate(zip(labels, axes.flat)):
        cm = confusion_matrix(y_true[:, i], y_pred[:, i])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["clean", label])
        disp.plot(ax=ax, colorbar=False)
        ax.set_title(label)
    plt.tight_layout()
    out = FIGURES_DIR / f"confusion_matrix_{model_name.lower().replace(' ', '_').replace('+', '').replace('-', '_')}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(out)
    return saved


def build_ovr_pipeline(base_clf, ovr_n_jobs: int = -1) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(**TFIDF_PARAMS)),
        ("clf", OneVsRestClassifier(base_clf, n_jobs=ovr_n_jobs)),
    ])


def fit_and_evaluate(name: str, pipeline: Pipeline, X_train, X_test, y_train, y_test, config: dict) -> dict:
    print(f"\n{'='*60}")
    print(f"  Training: {name}")
    print(f"{'='*60}")

    run = wandb.init(
        project=WANDB_PROJECT,
        name=name,
        config=config,
        reinit=True,
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Get probability estimates for AUC-ROC
    clf = pipeline.named_steps["clf"]
    if hasattr(clf, "predict_proba"):
        y_prob = pipeline.predict_proba(X_test)
    else:
        y_prob = pipeline.decision_function(X_test)
        # Normalise decision scores to [0,1] per column
        y_prob = (y_prob - y_prob.min(axis=0)) / (y_prob.max(axis=0) - y_prob.min(axis=0) + 1e-8)

    # Ensure 2-D arrays
    if y_prob.ndim == 1:
        y_prob = y_prob[:, np.newaxis]

    metrics = compute_metrics(y_test, y_pred, y_prob, LABEL_COLS)
    run.log(metrics)

    # Per-label classification reports
    for i, label in enumerate(LABEL_COLS):
        report = classification_report(y_test[:, i], y_pred[:, i], target_names=["clean", label], zero_division=0)
        print(f"\n  [{label}]\n{report}")

    print(f"\n  macro F1 : {metrics['macro_f1']:.4f}")
    print(f"  weighted F1: {metrics['weighted_f1']:.4f}")
    print(f"  macro AUC-ROC: {metrics['macro_auc_roc']:.4f}")

    # Confusion matrices
    cm_paths = save_confusion_matrices(y_test, y_pred, LABEL_COLS, name)
    for p in cm_paths:
        run.log({f"confusion_matrix/{name}": wandb.Image(str(p))})
        print(f"  Saved confusion matrix → {p}")

    run.finish()
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data…")
    df = pd.read_parquet(DATA_PATH)
    X = df["comment_text"].astype(str).to_numpy()
    y = df[LABEL_COLS].values.astype(int)

    print(f"  Rows: {len(X):,}  |  Labels: {LABEL_COLS}")
    print(f"  Label counts: { {l: int(y[:, i].sum()) for i, l in enumerate(LABEL_COLS)} }")

    # Stratified split on 'toxic' (most populated label, good proxy)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y[:, LABEL_COLS.index("toxic")],
    )
    print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    results: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Model 1 — TF-IDF (100K) + Logistic Regression
    # ------------------------------------------------------------------
    lr100_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(**TFIDF_PARAMS_LR_100K)),
        ("clf", OneVsRestClassifier(LogisticRegression(**LR_PARAMS), n_jobs=-1)),
    ])
    lr100_config = {"model": "LogisticRegression", "tfidf": TFIDF_PARAMS_LR_100K, "lr": LR_PARAMS}
    lr100_metrics = fit_and_evaluate(
        "TF-IDF + LR (100K)", lr100_pipeline, X_train, X_test, y_train, y_test, lr100_config
    )
    results["LR (100K)"] = lr100_metrics
    joblib.dump(lr100_pipeline, MODELS_DIR / "tfidf_lr.joblib")
    print(f"  Saved model → {MODELS_DIR / 'tfidf_lr.joblib'}")

    # ------------------------------------------------------------------
    # Model 2 — TF-IDF (20K) + Logistic Regression
    # ------------------------------------------------------------------
    lr20_pipeline = build_ovr_pipeline(LogisticRegression(**LR_PARAMS))
    lr20_config = {"model": "LogisticRegression", "tfidf": TFIDF_PARAMS, "lr": LR_PARAMS}
    lr20_metrics = fit_and_evaluate(
        "TF-IDF + LR (20K)", lr20_pipeline, X_train, X_test, y_train, y_test, lr20_config
    )
    results["LR (20K)"] = lr20_metrics
    joblib.dump(lr20_pipeline, MODELS_DIR / "tfidf_lr_20k.joblib")
    print(f"  Saved model → {MODELS_DIR / 'tfidf_lr_20k.joblib'}")

    # ------------------------------------------------------------------
    # Model 3 — TF-IDF (20K) + XGBoost
    # scale_pos_weight set per-label inside BalancedXGBClassifier.
    # OvR n_jobs=1: XGBoost already parallelises internally; running both causes OOM
    # ------------------------------------------------------------------

    class BalancedXGBClassifier(XGBClassifier):
        def fit(self, X, y, **kwargs):
            neg, pos = (y == 0).sum(), (y == 1).sum()
            self.set_params(scale_pos_weight=neg / max(pos, 1))
            return super().fit(X, y, **kwargs)

    xgb_pipeline = build_ovr_pipeline(BalancedXGBClassifier(**XGB_PARAMS), ovr_n_jobs=1)
    xgb_config = {"model": "XGBoost", "tfidf": TFIDF_PARAMS, "xgb": XGB_PARAMS}
    xgb_metrics = fit_and_evaluate(
        "TF-IDF + XGBoost (20K)", xgb_pipeline, X_train, X_test, y_train, y_test, xgb_config
    )
    results["XGB (20K)"] = xgb_metrics
    joblib.dump(xgb_pipeline, MODELS_DIR / "tfidf_xgb.joblib")
    print(f"  Saved model → {MODELS_DIR / 'tfidf_xgb.joblib'}")

    # ------------------------------------------------------------------
    # 3-way summary table
    # ------------------------------------------------------------------
    print(f"\n{'='*72}")
    print("  3-WAY COMPARISON (same 80/20 stratified split)")
    print(f"{'='*72}")

    rows = []
    for model_key, metrics in results.items():
        row: dict = {"Model": model_key}
        for label in LABEL_COLS:
            row[f"{label[:6]}_F1"] = round(metrics[f"{label}/f1"], 3)
        row["macro_F1"] = round(metrics["macro_f1"], 3)
        row["wtd_F1"] = round(metrics["weighted_f1"], 3)
        row["AUC_ROC"] = round(metrics["macro_auc_roc"], 3)
        rows.append(row)

    summary_df = pd.DataFrame(rows).set_index("Model")
    print(summary_df.to_string())

    comparison_run = wandb.init(
        project=WANDB_PROJECT,
        name="baseline_comparison_3way",
        reinit=True,
    )
    comparison_run.log({"summary_table": wandb.Table(dataframe=summary_df.reset_index())})
    comparison_run.finish()

    print(f"\nAll done. Models saved to {MODELS_DIR}/")


if __name__ == "__main__":
    main()
