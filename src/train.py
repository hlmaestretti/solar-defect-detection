import mlflow
import mlflow.sklearn
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


# If using Sklearn, delete the pytorch model and use this instead
def train_model(X_train, y_train, config):
    model_type = config["training"]["model_type"]
    mlflow.log_param("framework", "sklearn")
    mlflow.log_param("model_type", model_type)

    if model_type == "linear_regression":
        model = LinearRegression()
    elif model_type == "random_forest":
        model = RandomForestRegressor(
            random_state=config["training"]["random_state"]
        )
    else:
        raise ValueError(f"Unsupported sklearn model: {model_type}")

    model.fit(X_train, y_train)

    mlflow.sklearn.log_model(model, artifact_path="model")
    return model


# If using PyTorch, delete the sklearn model and use this instead
class SimpleMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)


def train_model(X_train, y_train, config):
    mlflow.log_param("framework", "pytorch")

    input_dim = X_train.shape[1]
    model = SimpleMLP(input_dim)

    lr = config["training"]["lr"]
    epochs = config["training"]["epochs"]

    mlflow.log_param("lr", lr)
    mlflow.log_param("epochs", epochs)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        optimizer.zero_grad()
        preds = model(X_train)
        loss = criterion(preds, y_train)
        loss.backward()
        optimizer.step()

        mlflow.log_metric("train_loss", loss.item(), step=epoch)

    mlflow.pytorch.log_model(model, artifact_path="model")
    return model
