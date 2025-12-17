import argparse
import yaml
import mlflow

from src.ingest import ingest_data
from src.preprocess import preprocess_data
from src.train import train_model
from src.evaluate import evaluate_model
from src.utils.mlflow_utils import init_mlflow


def run_pipeline(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    init_mlflow()
    mlflow.set_experiment(config["logging"]["experiment_name"])

    with mlflow.start_run():
        # Step 1: Ingest
        df = ingest_data(config)

        # Step 2: Preprocess
        X_train, X_test, y_train, y_test = preprocess_data(df, config)

        # Step 3: Train
        model = train_model(X_train, y_train, config)

        # Step 4: Evaluate
        metrics = evaluate_model(model, X_test, y_test)

        print("Pipeline completed.")
        print(metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="pipelines/config/default.yaml"
    )
    args = parser.parse_args()
    run_pipeline(args.config)
