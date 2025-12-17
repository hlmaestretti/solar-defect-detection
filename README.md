# MLOps Project Template

This repository provides a reusable, cloud-agnostic template for building 
production-ready machine learning systems.


## Features
- Modular data pipelines
- MLflow integration (pluggable backend)
- CI/CD templates
- Cloud deployment templates
- Monitoring + drift detection structure
- Reproducible configuration pattern

## Architecture Overview

This project is designed as a reusable MLOps platform template, separating
concerns across training, promotion, serving, CI/CD, and monitoring.

High-level flow:

1. Training pipeline ingests data and logs runs to MLflow
2. Evaluation metrics are recorded per run
3. Promotion logic compares metrics and manages model stages
4. FastAPI service loads models from the MLflow registry
5. CI/CD workflows orchestrate training and deployment
6. Monitoring detects drift and triggers retraining externally

## Framework Selection Philosophy (Template Design)

This template is intentionally designed to be edited, not dynamically configured.

For stages that are framework-specific (preprocessing, training, evaluation),
you may see multiple implementations of the same function (e.g., scikit-learn
and PyTorch) defined in the same file with identical function names.

### Why this design?

- This repository is a starter template, not a runtime framework.
- Real projects typically commit to one ML framework per service.
- Deleting unused code is clearer than maintaining runtime branching logic.
- The pipeline runner remains framework-agnostic and stable.


## Model Lifecycle

This project uses the MLflow Model Registry with the following lifecycle:

- **None** — newly trained models
- **Staging** — models that pass evaluation gates
- **Production** — models actively deployed

Model promotion is handled by a dedicated script (`src/promote_model.py`)
and is intentionally decoupled from training.

This allows:
- explicit promotion rules
- CI/CD-based gating
- safe rollback
- human-in-the-loop approvals


## Deployment (Template)

This repository includes a FastAPI + Cloud Run deployment scaffold.

The deployment is intentionally not runnable without project-specific
configuration. Before deploying, users must:

- specialize preprocessing, training, and evaluation code
- configure MLflow tracking and registry
- set MODEL_NAME and MODEL_STAGE environment variables
- build and push a container image

The provided Dockerfile and Cloud Run YAML define the expected
deployment contract, not a turn-key service.


## CI/CD (Template)

This repository includes skeleton GitHub Actions workflows that document the
intended CI/CD lifecycle:

- Continuous Integration:
  - run tests
  - train models
  - evaluate metrics
  - optionally promote models

- Continuous Deployment:
  - build a FastAPI service container
  - deploy to Cloud Run


## Monitoring & Drift Detection (Template)

This template includes scaffolding for monitoring and drift detection.

Drift detection is designed as a standalone process that:
- compares reference and current data
- produces a structured report
- triggers retraining externally if thresholds are exceeded

Monitoring is intentionally decoupled from training and deployment.



## How to Use This Template

1. Clone the repo:
   git clone https://github.com/hlmaestretti/mlops_template

2. Copy `.env.example` → `.env` and fill in your environment:
   - MLflow tracking server
   - Artifact store URI
   - Credentials

3. Decide which framework you will use (e.g., scikit-learn or PyTorch).
4. In each stage file (`preprocess.py`, `train.py`, `evaluate.py`):
   - Delete the unused implementation.
   - Keep the single function matching your framework.

5. Follow `docs/mlflow_setup.md` to configure MLflow for THIS project.

6. Run a test pipeline:
   python pipelines/run_pipeline.py

7. Deploy using:
   - FastAPI
   - Docker
   - Cloud Run or AWS ECS






