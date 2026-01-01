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
