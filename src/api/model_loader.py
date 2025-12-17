import mlflow
import os


def load_model():
    """
    Template model loader.

    Expected environment variables (to be set per project):
    - MODEL_NAME
    - MODEL_STAGE (e.g. Staging or Production)
    """

    model_name = os.getenv("MODEL_NAME", "example_model")
    model_stage = os.getenv("MODEL_STAGE", "Staging")

    model_uri = f"models:/{model_name}/{model_stage}"

    # This will fail unless MLflow is properly configured.
    return mlflow.pyfunc.load_model(model_uri)
