import mlflow
import torch
from sklearn.metrics import mean_squared_error, r2_score


# If using Sklearn, delete the pytorch model and use this instead
def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)

    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2", r2)

    return {"mse": mse, "r2": r2}


# If using PyTorch, delete the sklearn model and use this instead
def evaluate_model(model, X_test, y_test):
    with torch.no_grad():
        preds = model(X_test)

    mse = ((preds - y_test) ** 2).mean().item()
    r2 = 1 - mse / y_test.var().item()

    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2", r2)

    return {"mse": mse, "r2": r2}
