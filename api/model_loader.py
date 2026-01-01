'''
Model loading utilities for the EL defect detection service.

Responsibilities:
- Load the trained PyTorch model from the MLflow Model Registry using a
  registry URI 
- Move the model to the appropriate device (CPU or CUDA)
- Set the model to evaluation mode.
'''

import mlflow.pytorch
import torch


MODEL_URI = "models:/el_defect_classifier/Production"


def load_model():
    """
    Load the Production EL defect classifier from MLflow.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = mlflow.pytorch.load_model(MODEL_URI)
    model.to(device)
    model.eval()

    return model, device
