# Data role interview practice

## Project explanation

**1. Explain your capstone in 60 seconds.**  
I built an end-to-end churn prediction system for a telecom-style dataset. I validated 500 customer records, explored churn patterns, engineered business features, compared models, tuned a Random Forest with stratified cross-validation, selected a decision threshold from training folds, and evaluated on an untouched test set. I then saved the model and created Streamlit and FastAPI demos. I used recall, F1, ROC-AUC, and PR-AUC because the target was imbalanced, and I documented limitations and a transparent retention scenario.

**2. Why not rely only on accuracy?**  
With 10.6% churn, a model predicting everyone will stay would appear accurate but miss every churner. Recall, precision, F1, ROC-AUC, PR-AUC, and the confusion matrix show performance on the minority class.

**3. How did you avoid data leakage?**  
I split before fitting transformations, kept CustomerID and Churn out of features, placed feature engineering and preprocessing inside a scikit-learn Pipeline, and selected the threshold using out-of-fold training predictions.

**4. Why Random Forest?**  
It captures nonlinear relationships and interactions, supports class weighting, performs well on mixed tabular features after encoding, and provides global feature importance. I still compared it with simpler baselines.

**5. What would you improve with more time?**  
Collect more representative data, validate calibration, evaluate subgroups, add temporal validation, run an A/B retention experiment, monitor drift, and compare explainability methods such as permutation importance or SHAP.

## Common technical questions

6. Train vs validation vs test sets?  
Training fits parameters; validation/CV selects models and thresholds; the test set provides the final unbiased estimate.

7. Precision vs recall?  
Precision asks how many flagged customers truly churn; recall asks how many real churners were found.

8. What is cross-validation?  
Repeatedly train on several folds and validate on the held-out fold to estimate stability and guide selection.

9. Overfitting signs?  
Training performance far above validation/test performance, unstable fold scores, or overly complex models.

10. One-hot vs label encoding?  
One-hot avoids imposing order on nominal categories; label/ordinal encoding is suitable when categories have meaningful order or for a target label.

11. Standardization vs normalization?  
Standardization centers and scales by standard deviation; min-max normalization maps to a fixed range.

12. What is ROC-AUC?  
It measures ranking quality across classification thresholds: the probability a random positive is ranked above a random negative.

13. Why can PR-AUC be useful?  
It focuses on the positive class and is informative when positives are rare.

14. Explain a confusion matrix.  
It counts true negatives, false positives, false negatives, and true positives at a chosen threshold.

15. How would you deploy safely?  
Reproduce versions, validate inputs, secure endpoints, log predictions, monitor drift and delayed outcomes, use human review, and support rollback.

## Behavioral questions

16. Tell me about a problem you solved.  
Use STAR: dataset issue, validation approach, action, verified result.

17. How do you explain a model to a nontechnical person?  
Begin with the decision it supports, use a concrete example, explain tradeoffs, and state limitations.

18. Describe a mistake.  
Choose a real issue such as path/schema mismatch, explain diagnosis, fix, test added, and learning.

19. How do you prioritize learning?  
Connect learning to role requirements, build small projects, document them, and review gaps after feedback.

20. Why should we hire you?  
Emphasize practical projects, consistent learning, communication, careful validation, and readiness to grow in an entry-level role.
