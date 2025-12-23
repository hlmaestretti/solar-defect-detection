'''
Evaluation utilities for the EL defect multi-label classifier.

Responsibilities:
- Run inference on the validation loader without gradient tracking
- Convert logits to probabilities and compute multi-label predictions
- Compute and log:
  - validation macro recall 
  - per-class validation recall 
- Return a structured metric summary for downstream use

Notes:
- Ultra-rare classes may show zero recall due to insufficient data

'''

import torch
import mlflow
import yaml
from pathlib import Path
from sklearn.metrics import recall_score


# Load defect labels 
LABELS_PATH = Path("src/config/defect_labels.yaml")

with open(LABELS_PATH, "r") as f:
    DEFECT_LABELS = yaml.safe_load(f)["labels"]


# Evaluation
def evaluate_model(model, val_loader, config):
    """
    Evaluate a multi-label EL defect classification model.
    Computes per-class recall and macro recall.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).int()

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # Metrics
    per_class_recall = recall_score(
        y_true,
        y_pred,
        average=None,
        zero_division=0,
    )

    macro_recall = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    # MLflow logging
    mlflow.log_metric("val_macro_recall", macro_recall)

    for label, recall in zip(DEFECT_LABELS, per_class_recall):
        mlflow.log_metric(f"val_recall_{label}", recall)

    return {
        "macro_recall": macro_recall,
        "per_class_recall": dict(zip(DEFECT_LABELS, per_class_recall)),
    }
