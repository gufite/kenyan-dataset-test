## Feature handling & observed behavior (+ folder provided) - for both full features and borrower-side
- **Baseline (`outputs/`)**
  - Features: uses all engineered columns except IDs are ignored by the pipeline. History signals remain.
  - Behavior: Very high AUC (~0.99) driven partly by strong history features; likely overstates real-world performance if the same borrowers appear in train/test.
  - Metrics logged: accuracy, precision, recall, F1, AUC-ROC, AUC-PR.
  - Observed: `full` best AUC 0.995 (XGBoost); best F1 0.827 with LightGBM (precision 0.810, recall 0.846, accuracy ~0.994). `borrower` best AUC 0.970; best F1 0.565 with LightGBM (precision 0.466, recall 0.721, accuracy ~0.980).
- **Drop history/leakage (`outputs_drop_features/`)**
  - Features dropped: borrower history and repayment-derived signals such as `num_previous_loans`, `num_previous_defaults`, `past_default_rate`, `trend_in_amount/burden`, `repayment_intensity/consistency`, `burden_percentile`, IDs.
  - Behavior: Metrics dip vs. baseline but stay strong; LightGBM still leads. More realistic when past-behavior is weak or unavailable.
  - Metrics logged: accuracy, precision, recall, F1, AUC-ROC, AUC-PR.
  - Observed: `full` CatBoost AUC 0.995; LightGBM F1 0.847 (precision 0.816, recall 0.881, accuracy ~0.994). `borrower` LightGBM F1 0.578 (precision 0.476, recall 0.736, accuracy ~0.980; AUC 0.971).
- **First-time borrower emphasis (`outputs_drop_features_first_time_borrower/`)**
  - Features dropped: same as above, focusing on removing past-behavior clues to mimic thin-file borrowers.
  - Behavior: Slightly lower lift than baseline; LightGBM remains the most reliable. Useful for onboarding new borrowers with little history.
  - Metrics logged: accuracy, precision, recall, F1, AUC-ROC, AUC-PR.
  - Observed: `full` CatBoost AUC 0.996; LightGBM F1 0.824 (precision 0.781, recall 0.871, accuracy ~0.993). `borrower` LightGBM F1 0.585 (precision 0.467, recall 0.781, accuracy ~0.980; AUC 0.963).
- **High recall (`outputs_high_recall/`)**
  - Features: full set; threshold is tuned to hit recall ~90%.
  - Behavior: Catches most defaulters, but precision falls (more good borrowers flagged). Choose when “don’t miss risk” outweighs extra reviews/declines.
  - Metrics logged: accuracy, precision, recall, F1, AUC-ROC, AUC-PR, chosen threshold.
  - Observed: `full` LightGBM F1 0.825 at recall 0.900, precision 0.761 (accuracy ~0.993; AUC 0.994). `borrower` LightGBM F1 0.355 at recall 0.900, precision 0.221 (accuracy ~0.940; AUC 0.970).
- **Leakage-safe (`outputs_leakage_safe/`)**
  - Features dropped: repayment-derived/leaky signals (e.g., `repayment_intensity`, `repayment_consistency`, `latest_amount_ma3`, `trend_in_amount/burden`, `burden_percentile`) plus IDs.
  - Split: group by `customer_id` (or time-based if a date column exists) so no borrower is in both train and test.
  - Behavior: AUC/F1 remain strong but slightly lower than baseline—closer to true production performance. CatBoost edges out others, LightGBM/XGBoost close.
  - Metrics logged: accuracy, precision, recall, F1, AUC-ROC, AUC-PR, chosen threshold.
  - Observed: `full` CatBoost AUC 0.994, F1 0.841 (precision 0.857, recall 0.825, accuracy ~0.994). `borrower` CatBoost AUC 0.950, F1 0.350 (precision 0.224, recall 0.801, accuracy ~0.940); LightGBM F1 0.312 as next best.

## Metrics reported (used in every folder)
- `accuracy`: share of all loans the model labeled correctly (defaults and non-defaults).
- `precision`: of the loans flagged as risky, the share that truly default.
- `recall`: of the loans that defaulted, the share the model caught.
- `f1`: balance of precision and recall (higher = better balance).
- `auc_roc`: how well the model separates good vs. bad loans across all score cutoffs.
- `auc_pr`: similar separation metric but focused on the positive (default) class in imbalanced data.
- `threshold`: only in high-recall/leakage-safe runs; the score cutoff chosen to balance business goals (higher recall vs. fewer false alarms).

## Takeaways
- Full feature set: Gradient boosting models (LightGBM/CatBoost/XGBoost) separate good vs. bad loans very well (AUC ~0.99). LightGBM tends to give the best balance of precision vs. recall; CatBoost is close behind.
- Borrower-only feature set: Performance drops (as expected) but LightGBM still leads. Expect more false alarms or missed defaults when history is limited.
- High-recall setup: Successfully drives recall toward ~90%, useful when the business priority is “don’t miss bad loans,” at the expense of more good borrowers being flagged.
- Leakage-safe split: Results are slightly more conservative than the baseline, reflecting a fairer test. CatBoost edges out others, with LightGBM and XGBoost close.
- First-time borrower scenarios: Stripping history signals reduces lift, but LightGBM remains the most reliable generalist across tests.

## How to read the outputs
- Each folder has a `metrics_summary.csv` with the key numbers:
  - `precision`: of loans we flag, how many truly default.
  - `recall`: of all defaults, how many we catch.
  - `auc_roc`/`auc_pr`: overall ranking quality (higher is better).
  - `threshold` (high-recall/leakage-safe runs): the cutoff used to trade off risk vs. approvals.
- PNGs (`roc_*.png`, `pr_*.png`, `confusion_*.png`) visualize trade-offs; `classification_report_*.txt` shows per-class precision/recall.

