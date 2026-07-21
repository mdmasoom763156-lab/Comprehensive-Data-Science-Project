# Technical Documentation - Customer Churn Prediction & Retention Strategy

        ## 1. Project objective

        Build a reproducible, leakage-safe classification system that estimates customer churn probability, assigns an operational risk level, and supports targeted retention decisions.

        ## 2. Success criteria

        - Recall >= 0.80
        - ROC-AUC >= 0.85
        - F1 >= 0.65
        - Precision >= 0.50
        - Reproducible model artifact, prediction API, interactive frontend, tests, and documentation

        ## 3. Dataset and validation

        The supplied dataset contains 500 customer rows and 9 original columns. There are 53 churn cases (10.6%). All 53 observed churn cases have tenure of 12 months or less, an unusually strong educational-data pattern that helps explain the near-perfect ranking metrics and may not generalize. Validation requires the exact schema, unique IDs, nonnegative numeric fields, no missing values, and binary churn labels. The SHA-256 hash recorded in model metadata identifies the training snapshot.

        `CustomerID` is used only for traceability. It is never included in training. The target is excluded before preprocessing.

        ## 4. Architecture

        ```text
        Raw CSV -> schema validation -> stratified split -> feature engineering
                -> numeric impute/scale + category impute/one-hot
                -> model comparison -> Random Forest grid search
                -> training-fold threshold selection -> holdout evaluation
                -> joblib artifact -> FastAPI / Streamlit
        ```

        ## 5. Feature engineering

        Eight derived features are produced inside the pipeline: average monthly spend, charges gap, estimated annual charges, tenure squared, tenure-charge interaction, month-to-month flag, autopay flag, and new-customer flag. These transformations do not use the target.

        ## 6. Model development

        Baselines are Logistic Regression and Gradient Boosting. The selected Random Forest is tuned across estimators, depth, leaf size, and feature sampling using five-fold stratified GridSearchCV with ROC-AUC scoring. The best cross-validation ROC-AUC is 0.9937.

        Best parameters: `{"model__max_depth": 6, "model__max_features": "sqrt", "model__min_samples_leaf": 2, "model__n_estimators": 300}`

        Threshold `0.38` is selected from out-of-fold training probabilities, optimizing a weighted F1/recall score while requiring recall >= 0.80 and precision >= 0.50 where possible.

        ## 7. Holdout results

        - **Accuracy:** 0.9680
- **Balanced_Accuracy:** 0.9821
- **Precision:** 0.7647
- **Recall:** 1.0000
- **F1:** 0.8667
- **ROC_AUC:** 1.0000
- **PR_AUC:** 1.0000
- **Specificity:** 0.9643

        Confusion matrix: TN=108, FP=4, FN=0, TP=13.

        ### Model comparison

        | Model               |   Precision |   Recall |    F1 |   ROC_AUC |   PR_AUC |
|:--------------------|------------:|---------:|------:|----------:|---------:|
| Logistic Regression |       0.812 |    1.000 | 0.897 |     1.000 |    1.000 |
| Gradient Boosting   |       0.923 |    0.923 | 0.923 |     0.995 |    0.963 |
| Tuned Random Forest |       0.765 |    1.000 | 0.867 |     1.000 |    1.000 |

        All declared success targets are met: `{'Recall': True, 'ROC_AUC': True, 'F1': True, 'Precision': True}`.

        ## 8. Interpretation

        Top global Random Forest importances:

        - TenureSquared: 0.2295
- IsNewCustomer: 0.2242
- Tenure: 0.2107
- TenureMonthlyInteraction: 0.1047
- AverageMonthlySpend: 0.0714
- ChargesGap: 0.0482
- EstimatedAnnualCharges: 0.0224
- IsMonthToMonth: 0.0196

        Importance indicates model usage, not causal effect. Retention interventions should be evaluated with controlled experiments.

        ## 9. Deployment

        - `models/churn_pipeline.joblib`: trusted model package, threshold, feature schema, and risk boundaries.
        - `deployment/api.py`: validated `POST /predict` FastAPI endpoint and health endpoint.
        - `deployment/streamlit_app.py`: interactive web interface.
        - `deployment/Dockerfile`: reproducible basic container demonstration.
        - `src/predict.py`: reusable function and CLI.

        Scikit-learn warns that pickle/joblib-based artifacts should only be loaded from trusted sources and should use compatible environments. Version metadata is recorded alongside this artifact. Official reference: https://scikit-learn.org/stable/model_persistence.html

        ## 10. Testing and validation

        Automated tests cover dataset integrity, split isolation, target leakage prevention, artifact loading, declared metrics, repeatable inference, API validation, executed notebook state, reports, deck, screenshots, and transparent impact assumptions.

        ## 11. Monitoring plan

        Track input distributions, missing/invalid requests, risk-band volume, latency, errors, delayed label performance, probability calibration, subgroup metrics, business conversion, and offer cost. Trigger investigation for material drift, recall deterioration, or risk-band volume changes. Retrain only after validating new labels and comparing against the current model.

        ## 12. Limitations and responsible use

        - The educational dataset is small and may not represent a live customer population.
        - Random split performance may exceed future temporal performance.
        - No causal retention experiment was conducted.
        - Financial impact is scenario-based.
        - The model should prioritize human review, not deny service or make fully automated decisions.

        ## 13. Reproducibility

        Run `python -m src.train`, then `python -m pytest tests -q`. Exact dependency versions from the training environment are stored in `models/model_metadata.json`; installable minimum versions are listed in `requirements.txt`.
