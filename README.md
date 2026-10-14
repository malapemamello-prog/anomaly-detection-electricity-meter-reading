# Anomaly Detection in Electricity Meter Readings

A supervised, building-aware anomaly classifier for hourly electricity meter readings. It is trained on labelled buildings and applied to **unseen buildings** to flag suspicious readings (meter faults, stuck sensors, unusual equipment or occupancy behaviour) for follow-up investigation.

---



## Overview

Energy providers and facility managers cannot manually review millions of hourly readings. This project builds an automated detector that:

- handles a **severe class imbalance** (about 2% of readings are anomalous),
- uses **32 per-building time-series features**,
- validates on **whole held-out buildings** (no building appears in both training and validation),
- compares three imbalance-aware tree models (XGBoost, Balanced Random Forest, CatBoost) using F1 and PR-AUC.

## Data

| Attribute | Train set | Test set |
|-----------|-----------|----------|
| Rows / buildings | 1,749,494 / 200 | 1,800,567 / 206 (no overlap with train) |
| Columns | `building_id`, `timestamp`, `meter_reading`, `anomaly` | `row_id`, `building_id`, `timestamp`, `meter_reading` |
| Granularity | Hourly | Hourly |
| Duplicates | 0 | 0 |
| Missing readings | 107,653 (6.15%) | 94,864 (5.27%) |
| Anomalous readings | 37,296 (2.13%) | Unlabelled |

**Class imbalance:** a model predicting "normal" for every reading would score 97.87% accuracy while catching zero anomalies, so **F1 and PR-AUC** are used as the primary metrics instead of accuracy.

**Building-level variation:** anomaly prevalence is uneven. Six buildings exceed 5% and 14 are below 1%; the highest are building 1319 (8.82%), 1258 (8.5%) and 439 (6.47%).

## Methods

| Step | Method | Purpose / Detail |
|------|--------|------------------|
| 1. Cleaning | Timestamp parsing, per-building chronological sort | Required for lag and rolling features |
| 2. Missing values | Per-building forward fill, then backward fill | Preserves each building's own scale (consumption differs by orders of magnitude); no global mean/median imputation |
| 3. EDA | Class balance, building-level rates, hour-of-day and month patterns | Motivate metrics, grouped validation and cyclical encodings |
| 4. Feature engineering | 32 features built within each `building_id` | Prevents cross-building leakage (see table below) |
| 5. Validation | Building-grouped 70/30 split, cross-checked with 5-fold `GroupKFold` | Mimics scoring on unseen buildings |
| 6. Models | XGBoost, Balanced Random Forest, CatBoost | Each configured for class imbalance |
| 7. Thresholding | F1-optimal threshold from the precision-recall curve on validation data | Avoids the default 0.5 cut-off; test labels never used |
| 8. Prediction | Best model applied to the test set | Exported as `sample_submission.csv` (`row_id`, `anomaly`) |
| 9. Interpretation | Review of top-flagged buildings and point anomalies | Explain flags using the engineered features |



## Results

### Table 1: Validation Performance

| Model | Threshold | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|-------|-----------|-----------|--------|----|---------|--------|
| XGBoost | 0.876 | 0.851 | 0.655 | 0.740 | 0.933 | 0.773 |
| **Balanced Random Forest** (selected) | 0.947 | **0.969** | **0.727** | **0.830** | 0.937 | **0.824** |
| CatBoost | 0.921 | 0.835 | 0.522 | 0.643 | **0.942** | 0.690 |

### Table 2: Validation Confusion Matrices (from Figure 4 of the report)

| Model | True Negative | False Positive | False Negative | True Positive |
|-------|---------------|----------------|----------------|---------------|
| XGBoost | 341,514 | 907 | 2,726 | 5,167 |
| Balanced Random Forest | 342,237 | 184 | 2,159 | 5,734 |
| CatBoost | 341,609 | 812 | 3,771 | 4,122 |

### Table 3: Test-Set Predictions (Balanced Random Forest, threshold = 0.947)

| Class | Readings | Share |
|-------|----------|-------|
| Normal | 1,781,561 | 98.94% |
| Anomalous | 19,006 | 1.06% |

### Table 4: Top 10 Test Buildings by Predicted Anomalies



## Key Findings

- The **Balanced Random Forest** was the best model: F1 = 0.830, PR-AUC = 0.824, precision = 0.969 (about 97% of flagged readings were true anomalies on validation).
- CatBoost had the highest ROC-AUC but the lowest recall (0.522) and PR-AUC (0.690), so ROC-AUC alone was misleading under 2% prevalence.
- The model flagged 1.06% of test readings, lower than the 2.13% training rate. This fits a conservative, high-precision model, though a genuine difference in prevalence between building samples is also possible.
- Flagged anomalies are mostly **contextual**, meaning deviations from a building's own recent behaviour rather than global outliers. This supports the residual, rolling and rate-of-change feature design.

## Limitations

- **Imputation:** 6% of readings were forward/back-filled. Filled flat stretches can look like stuck-sensor patterns, and the backward fill uses future values.
- **Unused metadata:** the building attribute files (`train_features.csv`, `test_features.csv`) were not merged.
- **No test labels:** test-set performance cannot be verified; results rely on validation metrics only.
- **Single validation split:** one 70/30 building split (60 validation buildings). `GroupKFold` was a sanity check, but per-fold results are not reported.
- **Reused validation set:** the same data drives early stopping, threshold tuning and model selection, so metrics may be optimistic. There is no separate untouched hold-out.
- **Inconsistency:** the stated validation size (525,776 rows) does not match the confusion matrices in Figure 4 (350,314 rows).
- **Missed anomalies:** the selected model's recall is 0.727, so 27% of validation anomalies are missed. The high threshold (0.947) favours precision over recall.
- **Prevalence gap:** the predicted rate on test data (1.06%) is lower than the training rate (2.13%). Without labels, conservatism cannot be separated from a true prevalence difference.
- **Tuning not documented:** hyperparameter tuning is not described, and all features appear numeric, so CatBoost's categorical handling may add little.
- **Qualitative interpretation:** the point-anomaly review covers 10 readings, with no feature-importance/SHAP analysis and no ground truth.

## Future Work

- Add a separate hold-out set, or nested grouped cross-validation, and report per-fold results.
- Compare with an unsupervised baseline (e.g. Isolation Forest) and test alternative imputation or missing-value flags.
- Merge building attribute files and test whether metadata improves recall.
- Use feature importance / SHAP to back up the anomaly explanations.
- Tune the threshold to the operational cost of missed faults versus false alarms.



## Contributors

- **Mamello Malape**
- **Boipelo Letwaba**
