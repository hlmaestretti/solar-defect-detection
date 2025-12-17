# Monitoring & Drift Detection

This directory defines the monitoring and drift detection scaffold
for projects created from this template.

## What Is Monitored

- Input feature distributions
- Target distribution (optional)
- Prediction outputs (optional)

## What Is NOT Included

- Live dashboards
- Scheduled jobs
- Alerts
- Automatic retraining

## Intended Workflow

1. Periodically collect recent production data
2. Compare against reference data
3. Generate a drift report (JSON)
4. If drift is detected:
   - trigger the training pipeline via CI/CD
   - evaluate and promote models using registry rules

This separation ensures monitoring remains observable,
auditable, and safe.
