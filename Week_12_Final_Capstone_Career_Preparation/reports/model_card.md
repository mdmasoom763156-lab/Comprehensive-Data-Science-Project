# Model Card - Churn Risk Classifier

- **Model:** Tuned Random Forest
- **Purpose:** Prioritize customers for retention review
- **Training data:** 375 stratified rows from a 500-row educational dataset
- **Test data:** 125 untouched stratified rows
- **Target:** Churn (1 = left, 0 = stayed)
- **Threshold:** 0.38
- **Recall / Precision / F1:** 1.000 / 0.765 / 0.867
- **ROC-AUC / PR-AUC:** 1.000 / 1.000

## Intended use

Retention prioritization with human review in an educational/demo setting.

## Out-of-scope use

Credit, pricing, service denial, employment, or other high-impact automated decisions.

## Risks

Dataset shift, poor probability calibration, subgroup performance gaps, label delay, and mistaken causal interpretation.

## Safeguards

Input validation, human review, audit logs, subgroup monitoring, drift alerts, controlled intervention tests, environment pinning, and rollback.
