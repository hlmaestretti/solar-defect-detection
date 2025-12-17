# MLflow Setup Guide (Cloud-Agnostic & Template-Friendly)

This document explains how to configure MLflow Tracking, Artifact Storage, and optionally the Model Registry for ANY new project created from this template.

## Overview — What You Need to Configure


### 1. Tracking Server 

This server stores:
- params
- metrics
- run metadata
- model registry metadata

This can be one of the following:
- local SQLite
- remote SQL database (MySQL, PostgreSQL, Cloud SQL, RDS, etc.)
- Databricks MLflow (hosted)
- Managed MLflow (Azure, AWS, GCP platforms if applicable)

### 2. Artifact Store
This stores:
- model files
- plots
- datasets
- artifact saved by MLflow

This can be:
- Google Cloud Storage (GCS)
- AWS S3
- Azure Blob
- MinIO
- Local filesystem 

## Configure Your Environment
1. Make a copy of the .env file and fill in the needed parameters
```
cp .env.example .env
```

## Option A - Run MLflow Locally (Quick Start)
### 1. Create a local backend store
```
touch mlflow_backend/backend.db
```

### 2. Configure artifact storage (local folder)

In .env:
```
MLFLOW_TRACKING_URI=sqlite:///mlflow_backend/backend.db
MLFLOW_ARTIFACT_URI=./mlruns
```

### 3. Start the MLflow Server
```
mlflow server \
  --backend-store-uri sqlite:///mlflow_backend/backend.db \
  --default-artifact-root ./mlruns \
  --host 0.0.0.0 \
  --port 5000
```

Open MLflow UI:
```
http://localhost:5000
```

### 4. Run a quick test
```python
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")

with mlflow.start_run():
    mlflow.log_param("test_param", 1)
    mlflow.log_metric("accuracy", 0.95)
```

You should see:
- the run
- params
- metrics
- artifact folder created


## Option B - MLflow With GCP

### 1. Create a GCS bucket for artifacts
```
gsutil mb -l us-central1 gs://<your-bucket-name>
```

Add to .env:
```
MLFLOW_ARTIFACT_URI=gs://<your-bucket-name>
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service_account.json
```

Authenticate:
```
gcloud auth activate-service-account --key-file=/path/to/service_account.json
```

### 2. Choose the backend store

  #### a. Simple Approach — Local SQLite + GCS Artifact (Great for development on GCP without setting up Cloud SQL)

In .env:
```
MLFLOW_TRACKING_URI=sqlite:///mlflow_backend/backend.db
MLFLOW_ARTIFACT_URI=gs://<your-bucket-name>
```

Start the server:
```
mlflow server \
  --backend-store-uri sqlite:///mlflow_backend/backend.db \
  --default-artifact-root gs://<your-bucket-name> \
  --host 0.0.0.0 \
  --port 5000
```



#### b. Production Approach — Cloud SQL Backend + GCS Artifacts (For real MLOps deployments)

Your URI will look like:
```
postgresql://<user>:<password>@<cloudsql-ip>:5432/<database>
```

Add to .env:
```
MLFLOW_TRACKING_URI=postgresql://<user>:<password>@<ip>/<db>
MLFLOW_ARTIFACT_URI=gs://<your-bucket>
```

Start the MLflow server:
```
mlflow server \
  --backend-store-uri postgresql://<user>:<pass>@<ip>/<db> \
  --default-artifact-root gs://<bucket> \
  --host 0.0.0.0 \
  --port 5000 
```

## 3. Test MLflow Logging in GCP Mode
```python
from utils.mlflow_utils import init_mlflow
import mlflow

init_mlflow()

with mlflow.start_run(run_name="gcp-test"):
    mlflow.log_param("lr", 0.01)
    mlflow.log_metric("rmse", 4.2)
```

Verify:
- run exists
- metrics logged
- artifacts uploaded to GCS bucket
