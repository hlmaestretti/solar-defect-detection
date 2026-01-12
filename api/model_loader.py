'''
Model loading utilities for the EL defect detection service.

Responsibilities:
- Load the trained PyTorch model from the MLflow Model Registry using a
  registry URI 
- Move the model to the appropriate device (CPU or CUDA)
- Set the model to evaluation mode.
'''

import os
import mlflow
import mlflow.pytorch
import torch


def load_model():
    """
    Load the EL defect classifier from the MLflow Model Registry.
    """
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    model_uri = os.getenv("MODEL_URI")
    if not model_uri:
        model_name = os.getenv("MODEL_NAME", "el_defect_classifier")
        model_stage = os.getenv("MODEL_STAGE", "Production")
        model_uri = f"models:/{model_name}/{model_stage}"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = mlflow.pytorch.load_model(model_uri)
    model.to(device)
    model.eval()

    return model, device
