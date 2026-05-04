# Safeguard Studio — Key Decisions & Findings

## 2026-05-03 — EDA: 931 label inconsistencies corrected

931 rows in the Jigsaw train set had at least one sub-label (`severe_toxic`, `obscene`, `threat`, `insult`, or `identity_hate`) set to 1 while `toxic=0`. This is an annotation inconsistency — any comment carrying a sub-label is by definition toxic. `toxic` was set to 1 for all affected rows, raising the toxic rate from 9.58% to 10.17%. Corrected dataset saved to `data/processed/train_cleaned.parquet`.

## 2026-05-04 — Phase 1.3: DistilBERT fine-tuning — pipeline validated, full training deferred

`src/transformer_classifier.py` implements two-phase DistilBERT fine-tuning (distilbert-base-uncased, HuggingFace Trainer, BCEWithLogitsLoss + per-label pos_weight). Phase 1 validation run (5K training rows, 1 epoch) completed successfully on CPU:

| Label | Precision | Recall | F1 | AUC-ROC |
|---|---|---|---|---|
| toxic | 0.740 | 0.803 | 0.770 | 0.970 |
| severe_toxic | 0.136 | 0.961 | 0.238 | 0.984 |
| obscene | 0.546 | 0.911 | 0.683 | 0.982 |
| threat | 0.037 | 0.480 | 0.069 | 0.890 |
| insult | 0.485 | 0.892 | 0.628 | 0.979 |
| identity_hate | 0.094 | 0.722 | 0.166 | 0.953 |
| **macro F1** | | | **0.426** | |
| **macro AUC-ROC** | | | | **0.960** |

Even with 4% of training data and 1 epoch, macro AUC-ROC (0.960) is already close to the LR (100K) baseline (0.980) — pretrained representations carry significant weight. Low F1 on minority classes is a threshold-calibration effect from high pos_weight (threat: 335×, identity_hate: 110×); full training is expected to resolve this.

**Decision:** Full training (~3 epochs × 127K rows) requires GPU — estimated 28 hours on CPU vs ~35 min on T4. Deferred until GPU instance is available. Run `python src/transformer_classifier.py` on a GPU host; script auto-detects device and enables fp16 when CUDA is present.

## 2026-05-04 — Phase 1.2: Baseline classifier results (TF-IDF + LR vs XGBoost)

Three TF-IDF + OvR baselines trained on `train_cleaned.parquet` (80/20 stratified split on `toxic`, n=159,571). Class imbalance handled via `class_weight="balanced"` for LR and per-label `scale_pos_weight` for XGBoost.

| Model        | toxic | severe | obscene | threat | insult | id_hate | macro F1 | wtd F1 | AUC-ROC |
|-------------|-------|--------|---------|--------|--------|---------|----------|--------|---------|
| LR (100K)   | 0.751 | 0.426  | 0.764   | 0.435  | 0.683  | 0.367   | 0.571    | 0.707  | 0.980   |
| LR (20K)    | 0.729 | 0.395  | 0.743   | 0.372  | 0.657  | 0.314   | 0.535    | 0.682  | 0.975   |
| XGB (20K)   | 0.692 | 0.372  | 0.759   | 0.392  | 0.627  | 0.294   | 0.523    | 0.661  | 0.959   |

**Findings:**
- LR (100K) is the best baseline on every label except `obscene`, where XGB (20K) edges it out (0.759 vs 0.743 at 20K) — likely due to clearer lexical triggers exploited by trees.
- Feature count matters more for LR than XGB: LR loses 3.6 macro-F1 points from 100K→20K, while XGB at 20K only trails LR (20K) by 1.2 points.
- AUC-ROC is strong for both architectures (0.96–0.98), indicating good ranking ability; the gap is in threshold-level precision on minority classes.
- Minority classes (`severe_toxic`, `threat`, `identity_hate`) remain hard across all models — low positive support and high class imbalance drive F1 below 0.45 even with balancing.

**Engineering note:** XGBoost OvR requires `n_jobs=2` (not `-1`) on this 15 GB machine — histogram buffers scale with `n_features × n_threads` and OOM at 50K features with `-1`. LR is unaffected. Both are self-contained `joblib` pipelines (TF-IDF vectorizer embedded).

## 2026-05-03 — EDA: 42 rows truncated at 5,000 chars, left as-is

42 comments hit the 5,000-character source cap and are cut off mid-word or mid-sentence. Inspection of endings confirmed hard truncation (e.g. `SECURI`, `AR`, `BA`). All 42 are toxic content, so labels are unaffected by the missing tail. No correction applied — the truncation introduces minor text-feature noise but does not justify dropping or imputing rows.
