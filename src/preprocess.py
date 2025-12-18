import xml.etree.ElementTree as ET
from pathlib import Path
import yaml
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import mlflow


# Load defect labels (single source of truth)
LABELS_PATH = Path("src/config/defect_labels.yaml")

with open(LABELS_PATH, "r") as f:
    DEFECT_LABELS = yaml.safe_load(f)["labels"]

LABEL_TO_INDEX = {label: idx for idx, label in enumerate(DEFECT_LABELS)}


# XML parsing
def parse_xml_labels(xml_path):
    """
    Parse a Pascal VOC XML file and return a multi-hot label vector.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    label_vector = torch.zeros(len(DEFECT_LABELS), dtype=torch.float32)

    for obj in root.findall("object"):
        defect_name = obj.find("name").text
        if defect_name in LABEL_TO_INDEX:
            label_vector[LABEL_TO_INDEX[defect_name]] = 1.0

    return label_vector


# PyTorch Dataset
class ELDataset(Dataset):
    def __init__(self, images_dir, annotations_dir, transform=None):
        self.images_dir = Path(images_dir)
        self.annotations_dir = Path(annotations_dir)
        self.transform = transform

        self.image_ids = sorted([
            p.stem for p in self.annotations_dir.glob("*.xml")
            if (self.images_dir / f"{p.stem}.jpg").exists()
        ])

        if not self.image_ids:
            raise RuntimeError("No matching image/XML pairs found.")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]

        image_path = self.images_dir / f"{image_id}.jpg"
        xml_path = self.annotations_dir / f"{image_id}.xml"

        image = Image.open(image_path).convert("L")
        labels = parse_xml_labels(xml_path)

        if self.transform:
            image = self.transform(image)

        return image, labels


# Preprocess entry point
def preprocess_data(config):
    """
    Build train and validation dataloaders for EL defect classification.
    """
    images_dir = config["data"]["images_dir"]
    annotations_dir = config["data"]["annotations_dir"]

    batch_size = config["training"]["batch_size"]
    val_split = config["training"].get("val_split", 0.2)
    seed = config["training"].get("split_seed", 42)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    full_dataset = ELDataset(
        images_dir=images_dir,
        annotations_dir=annotations_dir,
        transform=transform,
    )

    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=generator,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # -------------------------------------------------
    # MLflow metadata
    # -------------------------------------------------
    mlflow.log_params({
        "num_images": len(full_dataset),
        "num_train_images": len(train_dataset),
        "num_val_images": len(val_dataset),
        "num_labels": len(DEFECT_LABELS),
        "image_size": 224,
        "batch_size": batch_size,
        "val_split": val_split,
        "split_seed": seed,
    })

    return train_loader, val_loader