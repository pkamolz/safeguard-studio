# Safeguard Studio — Key Decisions & Findings

## 2026-05-03 — EDA: 931 label inconsistencies corrected

931 rows in the Jigsaw train set had at least one sub-label (`severe_toxic`, `obscene`, `threat`, `insult`, or `identity_hate`) set to 1 while `toxic=0`. This is an annotation inconsistency — any comment carrying a sub-label is by definition toxic. `toxic` was set to 1 for all affected rows, raising the toxic rate from 9.58% to 10.17%. Corrected dataset saved to `data/processed/train_cleaned.parquet`.

## 2026-05-03 — EDA: 42 rows truncated at 5,000 chars, left as-is

42 comments hit the 5,000-character source cap and are cut off mid-word or mid-sentence. Inspection of endings confirmed hard truncation (e.g. `SECURI`, `AR`, `BA`). All 42 are toxic content, so labels are unaffected by the missing tail. No correction applied — the truncation introduces minor text-feature noise but does not justify dropping or imputing rows.
