# Credit Card Fraud Detection — Faculty Viva Questions and Answers

1. **What is the target, and what do its labels mean?**  
   `Class` is the target. `0` denotes a normal transaction and `1` denotes fraud; the source CSV is preserved.

2. **Why is accuracy insufficient here?**  
   Fraud is rare (492 of 284,807 source rows). A model can appear accurate by predicting normal for nearly every row, so precision, recall, F1, PR-AUC, and ROC-AUC are also evaluated.

3. **What does PR-AUC measure?**  
   Average precision summarizes the precision-recall tradeoff across thresholds and is informative for rare positive classes.

4. **How did preprocessing avoid leakage?**  
   The split is stratified, and the scaler is fitted on training data only. Duplicate removal is applied to derived data before splitting, not to the source file.

5. **Which model is used?**  
   Logistic Regression with `class_weight="balanced"` and the configured maximum iterations. This raises minority-class sensitivity without changing source labels.

6. **What is the quality gate?**  
   The configured gate requires average precision of at least 0.50. The measured value was 0.7194, so it passed.

7. **What does MLflow track and register?**  
   The project records a run with parameters, metrics, and model artifact, registers `Credit_Card_Fraud_Production_Model`, and assigns alias `production` to version 3.

8. **Why keep a local registry manifest as well?**  
   It makes the lifecycle decision inspectable from project files even when a team chooses not to use a remote MLflow server. Its version numbering is independent from MLflow's model version.

9. **How does the API identify a transaction?**  
   `POST /predict` accepts 30 numeric features in CSV order, excluding `Class`, and returns a class prediction and fraud probability for each record.

10. **What do the current metrics say about the decision threshold?**  
    Recall is 0.8737 while precision is 0.0564, indicating many false positives at the current decision threshold. Operational use would require threshold/cost analysis with stakeholders.

11. **What did failure injection verify?**  
    Five invalid-data/model conditions were detected. The tests use controlled, reversible input/artifact changes and restore the original state.

12. **Was Docker deployment verified?**  
    The model bundle and local API were verified, and a Dockerfile is prepared. Docker was unavailable, so image construction and container execution remain unverified.

