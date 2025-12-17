from fastapi import FastAPI
from src.api.schemas import PredictionRequest, PredictionResponse
from src.api.model_loader import load_model

app = FastAPI(
    title="ML Model API",
    version="1.0.0",
)

# Loaded at startup in real deployments
model = None


@app.on_event("startup")
def startup_event():
    global model
    model = load_model()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/version")
def version():
    return {"model_stage": "registry-based"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    # This assumes a numeric model.
    # Projects may replace this logic entirely.
    prediction = model.predict([request.inputs])[0]
    return PredictionResponse(prediction=float(prediction))
