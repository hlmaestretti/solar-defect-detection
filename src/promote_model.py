import mlflow
from mlflow.tracking import MlflowClient


MODEL_NAME = "el_defect_classifier"
METRIC_NAME = "val_macro_recall"


def get_latest_run_metric(experiment_name):
    client = MlflowClient()

    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )

    if not runs:
        raise ValueError("No runs found")

    run = runs[0]
    metric = run.data.metrics.get(METRIC_NAME)

    if metric is None:
        raise ValueError(
            f"Metric '{METRIC_NAME}' not found in latest run"
        )

    return run.info.run_id, metric


def get_production_metric(client):
    try:
        versions = client.get_latest_versions(
            MODEL_NAME,
            stages=["Production"]
        )
        if not versions:
            return None

        prod_run_id = versions[0].run_id
        run = client.get_run(prod_run_id)
        return run.data.metrics.get(METRIC_NAME)

    except Exception:
        return None


def promote_if_better(experiment_name):
    client = MlflowClient()

    run_id, new_metric = get_latest_run_metric(experiment_name)
    prod_metric = get_production_metric(client)

    print(f"New model {METRIC_NAME}: {new_metric}")
    print(f"Production model {METRIC_NAME}: {prod_metric}")

    # Higher recall is better
    if prod_metric is None or new_metric > prod_metric:
        print("Promoting model to Staging")

        mlflow.register_model(
            model_uri=f"runs:/{run_id}/model",
            name=MODEL_NAME,
        )

        latest_versions = client.get_latest_versions(
            MODEL_NAME,
            stages=["None"]
        )

        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=latest_versions[-1].version,
            stage="Staging",
            archive_existing_versions=False,
        )

    else:
        print("Model not promoted")


if __name__ == "__main__":
    promote_if_better(experiment_name="template_experiment")
