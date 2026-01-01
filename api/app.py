'''
FastAPI application for Solar Panel EL Defect Detection.


Endpoints:
- GET /health
  Simple liveness check to confirm the service is running.

- POST /predict
  Accepts a single EL image upload and returns:
    - per-defect probabilities
    - per-defect boolean predictions using class-specific thresholds
'''

from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import io
import torch
from torchvision import transforms
import yaml
from pathlib import Path

from src.api.model_loader import load_model
from src.api.schemas import PredictResponse


# -------------------------------------------------
# Load labels
# -------------------------------------------------
LABELS_PATH = Path("src/config/defect_labels.yaml")
with open(LABELS_PATH) as f:
    DEFECT_LABELS = yaml.safe_load(f)["labels"]


# -------------------------------------------------
# Thresholds (documented tradeoffs)
# -------------------------------------------------
THRESHOLDS = {
    "crack": 0.35,
    # all others default to 0.5
}


def get_threshold(label: str) -> float:
    return THRESHOLDS.get(label, 0.5)


# -------------------------------------------------
# Image preprocessing (must match training)
# -------------------------------------------------
IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])


# -------------------------------------------------
# App + model load
# -------------------------------------------------
app = FastAPI(title="Solar Panel EL Defect Detection")

model, device = load_model()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("L")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    tensor = IMAGE_TRANSFORM(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits).squeeze(0).cpu().tolist()

    probabilities = dict(zip(DEFECT_LABELS, probs))
    predictions = {
        label: probabilities[label] >= get_threshold(label)
        for label in DEFECT_LABELS
    }

    return PredictResponse(
        probabilities=probabilities,
        predictions=predictions,
    )
