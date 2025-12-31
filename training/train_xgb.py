import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
import mlflow
import mlflow.xgboost

df = pd.read_parquet("data/processed/processed_df.parquet")

X = df.drop(columns=["Class"])
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)



mlflow.set_tracking_uri("mlruns")
exp = mlflow.set_experiment("Fraud Detection Experiment")
with mlflow.start_run(run_name="XGBoost_Fraud_Model", experiment_id=exp.experiment_id):
    model = xgb.XGBClassifier(
        n_estimators=120,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        eval_metric="auc"
    )
    model.fit(X_train, y_train)

    y_pred_label = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:,1]

    ap_score = average_precision_score(y_test, y_pred_prob)
    auc_score = roc_auc_score(y_test, y_pred_prob)
    clf_report = classification_report(y_test, y_pred_label)

    mlflow.log_metric("val_ap", ap_score)
    mlflow.log_metric("val_auc", auc_score)
    # mlflow.log_text(f"clf_report: {clf_report}")
    mlflow.sklearn.log_model(model, artifact_path="xgboost_model")


    print(f"Validation Average Precision: {ap_score:.4f}")
    print(f"Validation Roc Auc Score: {auc_score:.4f}")
    print(f"Validation Classification Report: {clf_report}")