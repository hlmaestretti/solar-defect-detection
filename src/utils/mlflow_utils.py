import mlflow
from dotenv import load_dotenv
import os

def init_mlflow():
    load_dotenv()

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", None)
    artifact_uri = os.getenv("MLFLOW_ARTIFACT_URI", None)

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    if artifact_uri:
        # Not set directly in MLflow; included for instruction clarity
        pass

    return tracking_uri
