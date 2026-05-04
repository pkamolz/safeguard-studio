"""
DistilBERT fine-tuning for multi-label toxicity classification.

Two-phase run:
  Phase 1 (validation): 5K-row subset, 1 epoch  — confirms pipeline end-to-end
  Phase 2 (full):       127K train rows, 3 epochs

Loss:   BCEWithLogitsLoss with per-label pos_weight (handles class imbalance)
Output: models/distilbert_toxicity/
W&B:    project 'safeguard-studio'
"""

import os
import warnings
from pathlib import Path

import dotenv
import joblib
import numpy as np
import pandas as pd
import torch
import wandb
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)

warnings.filterwarnings("ignore")
dotenv.load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_PATH     = Path("data/processed/train_cleaned.parquet")
MODELS_DIR    = Path("models")
WANDB_PROJECT = "safeguard-studio"
MODEL_NAME    = "distilbert-base-uncased"
LABEL_COLS    = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
RANDOM_STATE  = 42
TEST_SIZE     = 0.2

LR           = 2e-5
BATCH_SIZE   = 16
EVAL_BATCH   = 32
MAX_LENGTH   = 128
EPOCHS       = 3
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
THRESHOLD    = 0.5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.environ["WANDB_PROJECT"] = WANDB_PROJECT


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ToxicityDataset(Dataset):
    def __init__(self, encodings: dict, labels: np.ndarray):
        self.input_ids      = encodings["input_ids"]
        self.attention_mask = encodings["attention_mask"]
        self.labels         = labels.astype(np.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_ids":      torch.tensor(self.input_ids[idx],      dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_mask[idx], dtype=torch.long),
            "labels":         torch.tensor(self.labels[idx],          dtype=torch.float),
        }


# ---------------------------------------------------------------------------
# Custom Trainer — BCEWithLogitsLoss with per-label pos_weight
# ---------------------------------------------------------------------------

class WeightedTrainer(Trainer):
    def __init__(self, pos_weight: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight = pos_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.pop("labels")
        outputs = model(**inputs)
        loss_fn = torch.nn.BCEWithLogitsLoss(
            pos_weight=self.pos_weight.to(outputs.logits.device)
        )
        loss = loss_fn(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def make_compute_metrics():
    def compute_metrics(eval_pred: EvalPrediction) -> dict:
        logits = eval_pred.predictions
        labels = eval_pred.label_ids.astype(int)
        probs  = torch.sigmoid(torch.from_numpy(logits)).numpy()
        preds  = (probs >= THRESHOLD).astype(int)

        m: dict = {}
        for i, lbl in enumerate(LABEL_COLS):
            m[f"{lbl}/precision"] = precision_score(labels[:, i], preds[:, i], zero_division=0)
            m[f"{lbl}/recall"]    = recall_score(labels[:, i], preds[:, i], zero_division=0)
            m[f"{lbl}/f1"]        = f1_score(labels[:, i], preds[:, i], zero_division=0)
            try:    m[f"{lbl}/auc_roc"] = roc_auc_score(labels[:, i], probs[:, i])
            except: m[f"{lbl}/auc_roc"] = float("nan")

        m["macro_f1"]    = f1_score(labels, preds, average="macro",    zero_division=0)
        m["weighted_f1"] = f1_score(labels, preds, average="weighted", zero_division=0)
        try:    m["macro_auc_roc"] = roc_auc_score(labels, probs, average="macro", multi_class="ovr")
        except: m["macro_auc_roc"] = float("nan")
        return m

    return compute_metrics


# ---------------------------------------------------------------------------
# Tokenise helper
# ---------------------------------------------------------------------------

def tokenize(tokenizer, texts: np.ndarray) -> dict:
    return tokenizer(
        list(texts),
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="np",
    )


# ---------------------------------------------------------------------------
# Training runner
# ---------------------------------------------------------------------------

def run_training(
    X_train:    np.ndarray,
    X_test:     np.ndarray,
    y_train:    np.ndarray,
    y_test:     np.ndarray,
    run_name:   str,
    epochs:     int,
    output_dir: Path,
) -> tuple[dict, Trainer, AutoTokenizer]:
    print(f"\n{'='*64}")
    print(f"  {run_name}")
    print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}  |  Epochs: {epochs}")
    print(f"  Device: {DEVICE}")
    print(f"{'='*64}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("  Tokenising train set…")
    train_enc = tokenize(tokenizer, X_train)
    print("  Tokenising test set…")
    test_enc  = tokenize(tokenizer, X_test)

    train_ds = ToxicityDataset(train_enc, y_train)
    test_ds  = ToxicityDataset(test_enc,  y_test)

    pos_weight = torch.tensor(
        (y_train == 0).sum(0) / np.maximum((y_train == 1).sum(0), 1),
        dtype=torch.float32,
    )
    print(f"  pos_weight: { {l: round(float(w), 1) for l, w in zip(LABEL_COLS, pos_weight)} }")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_COLS),
        problem_type="multi_label_classification",
        ignore_mismatched_sizes=True,
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH,
        learning_rate=LR,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        report_to="wandb",
        run_name=run_name,
        logging_steps=50,
        fp16=(DEVICE == "cuda"),
        dataloader_num_workers=0,
        label_names=["labels"],
    )

    trainer = WeightedTrainer(
        pos_weight=pos_weight,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=make_compute_metrics(),
        processing_class=tokenizer,
    )

    trainer.train()

    eval_results = trainer.evaluate()
    metrics = {k.replace("eval_", ""): v for k, v in eval_results.items()}
    return metrics, trainer, tokenizer


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Device: {DEVICE}")
    if DEVICE == "cpu":
        print("WARNING: No GPU detected. Validation run ~10-15 min; full run will be very slow on CPU.")

    # Load and split — identical seed/stratify to baselines
    print("\nLoading data…")
    df = pd.read_parquet(DATA_PATH)
    X  = df["comment_text"].astype(str).to_numpy()
    y  = df[LABEL_COLS].values.astype(int)

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE,
        stratify=y[:, LABEL_COLS.index("toxic")],
    )
    print(f"  Full train: {len(X_train_full):,}  |  Test: {len(X_test):,}")

    # ------------------------------------------------------------------
    # Phase 1 — Validation run: 5K rows, 1 epoch
    # ------------------------------------------------------------------
    print("\n─── Phase 1: Validation run (5K rows, 1 epoch) ───")
    rng     = np.random.default_rng(RANDOM_STATE)
    val_idx = rng.choice(len(X_train_full), size=5_000, replace=False)

    val_metrics, _, _ = run_training(
        X_train_full[val_idx], X_test,
        y_train_full[val_idx], y_test,
        run_name="distilbert-validation-5k",
        epochs=1,
        output_dir=MODELS_DIR / "distilbert_validation",
    )
    print(f"\n  Pipeline OK — val macro F1: {val_metrics.get('macro_f1', 0):.4f}")

    # ------------------------------------------------------------------
    # Phase 2 — Full training: all rows, 3 epochs
    # ------------------------------------------------------------------
    print("\n─── Phase 2: Full training (all rows, 3 epochs) ───")
    full_dir = MODELS_DIR / "distilbert_toxicity"

    full_metrics, trainer, tokenizer = run_training(
        X_train_full, X_test, y_train_full, y_test,
        run_name="distilbert-full",
        epochs=EPOCHS,
        output_dir=full_dir,
    )

    trainer.save_model(str(full_dir))
    tokenizer.save_pretrained(str(full_dir))
    print(f"\n  Model saved → {full_dir}")

    # ------------------------------------------------------------------
    # Comparison table: DistilBERT vs LR (100K) baseline
    # ------------------------------------------------------------------
    print("\nBuilding comparison table…")
    lr_pipeline = joblib.load(MODELS_DIR / "tfidf_lr.joblib")
    lr_pred     = lr_pipeline.predict(X_test)
    lr_prob     = lr_pipeline.predict_proba(X_test)

    # DistilBERT predictions from best checkpoint
    test_enc = tokenize(tokenizer, X_test)
    test_ds  = ToxicityDataset(test_enc, y_test)
    pred_out = trainer.predict(test_ds)
    db_probs = torch.sigmoid(torch.from_numpy(pred_out.predictions)).numpy()
    db_preds = (db_probs >= THRESHOLD).astype(int)

    def metrics_row(name: str, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
        r: dict = {"Model": name}
        for i, lbl in enumerate(LABEL_COLS):
            r[f"{lbl[:6]}_F1"] = round(f1_score(y_test[:, i], y_pred[:, i], zero_division=0), 3)
        r["macro_F1"] = round(f1_score(y_test, y_pred, average="macro",    zero_division=0), 3)
        r["wtd_F1"]   = round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 3)
        try:    r["AUC_ROC"] = round(roc_auc_score(y_test, y_prob, average="macro", multi_class="ovr"), 3)
        except: r["AUC_ROC"] = float("nan")
        return r

    summary_df = pd.DataFrame([
        metrics_row("LR (100K)",  lr_pred,  lr_prob),
        metrics_row("DistilBERT", db_preds, db_probs),
    ]).set_index("Model")

    print(f"\n{'='*72}")
    print("  COMPARISON: DistilBERT vs LR (100K) Baseline")
    print(f"{'='*72}")
    print(summary_df.to_string())

    comp_run = wandb.init(
        project=WANDB_PROJECT, name="distilbert_vs_baseline", reinit=True
    )
    comp_run.log({"comparison_table": wandb.Table(dataframe=summary_df.reset_index())})
    comp_run.finish()

    print("\nAll done.")


if __name__ == "__main__":
    main()
