import pandas as pd
import mlflow

def ingest_data(config):
    input_path = config["data"]["input_path"]

    df = pd.read_csv(input_path)
    mlflow.log_param("input_rows", len(df))

    return df
