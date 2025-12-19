import argparse
import yaml
import mlflow
from pathlib import Path

from src.preprocess import preprocess_data
from src.train import train_model
from src.evaluate import evaluate_model


REQUIRED_CONFIG_KEYS = [
    "data",
    "training",
]


def validate_config(config: dict):
    for key in REQUIRED_CONFIG_KEYS:
        if key not in config:
            raise ValueError(f"Missing required config section: '{key}'")


def run_pipeline(config_path: str):
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load + validate config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    validate_config(config)

    experiment_name = config.get(
        "experiment_name",
        "el_defect_detection",
    )

    # MLflow setup
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(
        run_name="training_pipeline",
        tags={
            "pipeline": "training",
            "task": "multilabel_el_defect_classification",
            "dataset": "PVEL-AD",
        },
    ):
        # Log full config for reproducibility
        mlflow.log_artifact(
            local_path=str(config_path),
            artifact_path="config",
        )

        # Preprocessing
        train_loader, val_loader = preprocess_data(config)

        # Training
        model = train_model(
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
        )

        # Evaluation
        evaluate_model(
            model=model,
            val_loader=val_loader,
            config=config,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run EL defect detection training pipeline"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to pipeline config YAML",
    )

    args = parser.parse_args()
    run_pipeline(args.config)
