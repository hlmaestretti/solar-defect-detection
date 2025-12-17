import mlflow
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# If using Sklearn, delete the PyTorch implementation below
def preprocess_data(df, config):
    target = config["data"]["target_column"]
    test_size = config["preprocessing"]["test_size"]

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=config["training"]["random_state"],
    )

    mlflow.log_param("test_size", test_size)

    if config["preprocessing"].get("scaling", False):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        mlflow.log_param("scaling", True)
    else:
        mlflow.log_param("scaling", False)

    return X_train, X_test, y_train, y_test


# If using PyTorch, delete the Sklearn implementation above
def preprocess_data(df, config):
    target = config["data"]["target_column"]
    test_size = config["preprocessing"]["test_size"]

    X = df.drop(columns=[target]).values
    y = df[target].values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=config["training"]["random_state"],
    )

    mlflow.log_param("test_size", test_size)

    if config["preprocessing"].get("scaling", False):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        mlflow.log_param("scaling", True)
    else:
        mlflow.log_param("scaling", False)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    return X_train, X_test, y_train, y_test
