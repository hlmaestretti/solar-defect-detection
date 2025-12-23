'''
Inference utilities for EL defect multi-label classification.

Responsibilities:
- Load a trained PyTorch model from MLflow 
- Apply the same image preprocessing as training 
- Run single-image inference and return a label -> probability mapping for all defect types

Notes:
- This module does not apply thresholds by default; it returns probabilities to allow
  downstream systems to choose class-specific thresholds or operating points.
- Label ordering is defined by src/config/defect_labels.yaml.

'''

import torch
import yaml
from pathlib import Path
from PIL import Image
from torchvision import transforms
import mlflow


# Load defect labels (single source of truth)
LABELS_PATH = Path("src/config/defect_labels.yaml")

with open(LABELS_PATH, "r") as f:
    DEFECT_LABELS = yaml.safe_load(f)["labels"]


# Image preprocessing (must match training)
_IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])


# Model loading
def load_model(
    model_uri: str,
    device: str | None = None,
):
    """
    Load a PyTorch model from MLflow or local path.

    Example model_uri:
      - runs:/<run_id>/model
      - models:/el_defect_classifier/Production
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = mlflow.pytorch.load_model(model_uri)
    model.to(device)
    model.eval()

    return model, device


# Prediction
def predict_image(
    model,
    device,
    image_path: str | Path,
):
    """
    Run multi-label inference on a single EL image.
    Returns label → probability mapping.
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path).convert("L")
    tensor = _IMAGE_TRANSFORM(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits).squeeze(0)

    predictions = {
        label: float(prob)
        for label, prob in zip(DEFECT_LABELS, probs.cpu().tolist())
    }

    return predictions
