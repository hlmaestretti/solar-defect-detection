import yaml
import torch

DEFECT_COUNTS = {
    "finger": 1506,
    "crack": 969,
    "black_core": 1011,
    "thick_line": 775,
    "horizontal_dislocation": 266,
    "short_circuit": 492,
    "vertical_dislocation": 34,
    "star_crack": 123,
    "printing_error": 2,
    "corner": 7,
    "fragment": 7,
    "scratch": 3,
}

TOTAL_IMAGES = 4500

with open("src/config/defect_labels.yaml") as f:
    labels = yaml.safe_load(f)["labels"]

weights = []
for label in labels:
    p = DEFECT_COUNTS[label]
    w = ((TOTAL_IMAGES - p) / p) ** 0.5
    weights.append(w)

weights_tensor = torch.tensor(weights, dtype=torch.float32)

print("Class weights:")
for l, w in zip(labels, weights):
    print(f"{l:25s}: {w:.2f}")

print("\nPyTorch tensor:")
print(weights_tensor)