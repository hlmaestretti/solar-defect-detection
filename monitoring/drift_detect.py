"""
Lightweight drift detection for EL image inputs.

This script compares summary statistics derived from grayscale pixel
intensities between a reference dataset (training) and a current dataset
(recent production). It emits a JSON report with feature-level z-scores.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
FEATURE_NAMES = [
    "mean_intensity",
    "std_intensity",
    "p10_intensity",
    "p50_intensity",
    "p90_intensity",
]


def list_image_paths(image_dir: Path) -> list[Path]:
    if not image_dir.exists():
        raise FileNotFoundError(f"Directory not found: {image_dir}")

    paths = [
        p for p in image_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if not paths:
        raise ValueError(f"No images found under: {image_dir}")

    return sorted(paths)


def extract_image_features(image_path: Path) -> dict:
    image = Image.open(image_path).convert("L")
    arr = np.asarray(image, dtype=np.float32)

    p10, p50, p90 = np.percentile(arr, [10, 50, 90]).astype(float)

    return {
        "mean_intensity": float(arr.mean()),
        "std_intensity": float(arr.std()),
        "p10_intensity": float(p10),
        "p50_intensity": float(p50),
        "p90_intensity": float(p90),
    }


def load_dataset_features(
    image_dir: Path,
    max_images: int | None = None,
    seed: int = 42,
) -> list[dict]:
    paths = list_image_paths(image_dir)

    if max_images is not None and len(paths) > max_images:
        rng = random.Random(seed)
        paths = rng.sample(paths, max_images)

    return [extract_image_features(path) for path in paths]


def summarize_features(feature_rows: list[dict]) -> dict:
    summary = {}
    for name in FEATURE_NAMES:
        values = np.array([row[name] for row in feature_rows], dtype=np.float32)
        summary[name] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
        }
    return summary


def calculate_drift(reference_summary: dict, current_summary: dict, threshold: float) -> dict:
    z_scores = {}
    eps = 1e-6

    for name in FEATURE_NAMES:
        ref_mean = reference_summary[name]["mean"]
        ref_std = reference_summary[name]["std"]
        cur_mean = current_summary[name]["mean"]

        denom = max(ref_std, eps)
        z_scores[name] = abs(cur_mean - ref_mean) / denom

    drift_score = float(np.mean(list(z_scores.values()))) if z_scores else 0.0
    drift_detected = drift_score >= threshold

    return {
        "drift_detected": drift_detected,
        "drift_score": drift_score,
        "feature_z_scores": z_scores,
    }


def write_report(report: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="EL image drift detection")
    parser.add_argument("--reference-dir", required=True, help="Path to reference images")
    parser.add_argument("--current-dir", required=True, help="Path to current images")
    parser.add_argument("--output", default="monitoring/drift_report.json")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    reference_dir = Path(args.reference_dir)
    current_dir = Path(args.current_dir)
    output_path = Path(args.output)

    reference_rows = load_dataset_features(reference_dir, args.max_images, args.seed)
    current_rows = load_dataset_features(current_dir, args.max_images, args.seed)

    reference_summary = summarize_features(reference_rows)
    current_summary = summarize_features(current_rows)

    drift = calculate_drift(reference_summary, current_summary, args.threshold)

    report = {
        "reference_dir": str(reference_dir),
        "current_dir": str(current_dir),
        "reference_count": len(reference_rows),
        "current_count": len(current_rows),
        "threshold": args.threshold,
        "features": FEATURE_NAMES,
        "summary": {
            "reference": reference_summary,
            "current": current_summary,
        },
        **drift,
    }

    write_report(report, output_path)


if __name__ == "__main__":
    main()
