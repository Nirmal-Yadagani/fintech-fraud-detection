import mlflow

def log_metrics_to_mlflow(
    roc_auc: float,
    pr_auc: float,
    recall: float,
    precision: float,
    fbeta: float,
    tn: int,
    fp: int,
    fn: int,
    tp: int,
    log_prefix: str = ""
):
    """
    Logs fraud detection training metrics to MLflow.

    Args:
        roc_auc: ROC-AUC score
        pr_auc: Precision-Recall AUC score
        recall: Fraud recall (TPR)
        precision: Fraud precision (PPV)
        fbeta: F-beta score (β=2 or 3 recommended)
        tn, fp, fn, tp: Confusion matrix values
        log_prefix: Optional prefix like "val_" or "test_" for metric names
    """
    prefix = f"{log_prefix}_" if log_prefix else ""

    mlflow.log_metric(f"{prefix}roc_auc", roc_auc)
    mlflow.log_metric(f"{prefix}pr_auc", pr_auc)
    mlflow.log_metric(f"{prefix}recall", recall)
    mlflow.log_metric(f"{prefix}precision", precision)
    mlflow.log_metric(f"{prefix}fbeta_score", fbeta)

    # Confusion matrix values
    mlflow.log_metric(f"{prefix}true_negatives", tn)
    mlflow.log_metric(f"{prefix}false_positives", fp)
    mlflow.log_metric(f"{prefix}false_negatives", fn)
    mlflow.log_metric(f"{prefix}true_positives", tp)

    # Derived rates
    fpr = fp / (fp + tn + 1e-9)
    fnr = fn / (fn + tp + 1e-9)

    mlflow.log_metric(f"{prefix}false_positive_rate", fpr)
    mlflow.log_metric(f"{prefix}false_negative_rate", fnr)

    # Optional combined logging label
    if log_prefix:
        mlflow.set_tag("metrics_logged", log_prefix)
    else:
        mlflow.set_tag("metrics_logged", "final")

    print("Fraud metrics successfully logged to MLflow")
