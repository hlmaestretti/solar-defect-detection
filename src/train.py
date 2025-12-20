import torch
import torch.nn as nn
import mlflow
import mlflow.pytorch
from sklearn.metrics import recall_score
import copy

# CNN Model Definition
class ELDefectCNN(nn.Module):
    """
    Custom CNN for multi-label EL defect classification.
    Designed for grayscale images and industrial texture defects.
    """

    def __init__(self,  num_labels: int, blocks=3, init_channels=32, kernel_size=3, stride=1, padding=1):
        super().__init__()

        layers = []
        
        layers.append(nn.Conv2d(1, init_channels, kernel_size=kernel_size, padding=padding))
        layers.append(nn.BatchNorm2d(init_channels))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(2))

        curr_n = init_channels
        for _ in range(blocks):
            layers.append(nn.Conv2d(curr_n, 2*curr_n, kernel_size=kernel_size, padding=padding, stride=stride))
            layers.append(nn.BatchNorm2d(2*curr_n))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool2d(2))
            
            curr_n = 2*curr_n 

        self.model = nn.Sequential(*layers)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(curr_n, num_labels)

    def forward(self, x):
        x = self.model(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits


def train_model(train_loader, val_loader, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Config
    cfg = config["training"]
    num_labels = cfg["num_labels"]
    epochs = cfg["epochs"]
    lr = cfg["lr"]
    weight_decay = cfg["weight_decay"]

    # CNN params
    blocks = cfg["blocks"]
    init_channels = cfg["init_channels"]
    kernel_size = cfg["kernel_size"]
    stride = cfg["stride"]
    padding = cfg["padding"]

    # Early stopping params
    es_cfg = cfg.get("early_stopping", {})
    early_stop_enabled = es_cfg.get("enabled", False)
    patience = es_cfg.get("patience", 10)
    min_delta = es_cfg.get("min_delta", 0.0)

    # Model
    model = ELDefectCNN(
        num_labels=num_labels,
        blocks=blocks,
        init_channels=init_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
    ).to(device)

    # Optimizer & Loss
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    class_weights = cfg.get("class_weights")

    if class_weights is not None:
        pos_weight = torch.tensor(class_weights, device=device, dtype=torch.float32)

        # Clamp extremely large weights
        max_w = cfg.get("max_pos_weight", 50.0)
        pos_weight = torch.clamp(pos_weight, max=max_w)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()
        
    # MLflow params
    mlflow.log_params({
        "model_type": "ELDefectCNN",
        "num_labels": num_labels,
        "epochs": epochs,
        "lr": lr,
        "weight_decay": weight_decay,
        "blocks": blocks,
        "init_channels": init_channels,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "early_stopping_enabled": early_stop_enabled,
        "early_stopping_patience": patience,
        "early_stopping_min_delta": min_delta,
    })

    print_every = max(1, epochs // 20)

    # Early stopping state
    best_val_recall = -float("inf")
    best_epoch = 0
    best_weights = None
    epochs_without_improve = 0

    # Training Loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        y_true, y_pred = [], []

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            preds = (torch.sigmoid(logits) > 0.5).int()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

        train_loss /= len(train_loader)
        train_recall = recall_score(
            y_true, y_pred, average="macro", zero_division=0
        )

        # VALIDATION
        model.eval()
        val_loss = 0.0
        y_true, y_pred = [], []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                logits = model(images)
                loss = criterion(logits, labels)
                val_loss += loss.item()

                preds = (torch.sigmoid(logits) > 0.5).int()
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        val_loss /= len(val_loader)
        val_recall = recall_score(
            y_true, y_pred, average="macro", zero_division=0
        )

        # MLflow metrics
        mlflow.log_metrics({
            "train_loss": train_loss,
            "train_macro_recall": train_recall,
            "val_loss": val_loss,
            "val_macro_recall": val_recall,
        }, step=epoch)

        # Early stopping check
        improved = val_recall > best_val_recall + min_delta

        if improved:
            best_val_recall = val_recall
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1

        if epoch % print_every == 0 or epoch == epochs - 1:
            print(
                f"Epoch [{epoch+1:>3}/{epochs}] | "
                f"Train Loss: {train_loss:.4f}, "
                f"Train Recall: {train_recall:.4f} | "
                f"Val Loss: {val_loss:.4f}, "
                f"Val Recall: {val_recall:.4f}"
            )

        if early_stop_enabled and epochs_without_improve >= patience:
            print(
                f"\nEarly stopping triggered at epoch {epoch+1}. "
                f"Best val recall: {best_val_recall:.4f} "
                f"(epoch {best_epoch+1})"
            )
            mlflow.log_metric("early_stop_epoch", epoch)
            break


    # Restore best model
    if best_weights is not None:
        model.load_state_dict(best_weights)

    # Save model artifact
    mlflow.pytorch.log_model(
        model,
        artifact_path="model",
        registered_model_name=None,
    )

    return model