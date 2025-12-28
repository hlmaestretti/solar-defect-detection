# Solar Defect Detection (EL Images)

Multi-label electroluminescence (EL) defect classification for solar modules. A custom PyTorch pipeline trains from scratch, logs to MLflow, and can promote improved models via the MLflow Model Registry.

## Project status
- End-to-end training pipeline (`pipelines/run_pipeline.py`) that logs configs, params, metrics, and the trained model to MLflow.
- Deterministic train/validation split with torchvision transforms and per-run dataset stats.
- Custom CNN (ELDefectCNN) optimized for grayscale EL imagery; uses BCEWithLogitsLoss, AdamW, optional class weights, and early stopping on macro recall.
- Evaluation computes macro recall plus per-class recall; promotion script compares `val_macro_recall` against Production.

## Dataset and labels
- Input: grayscale EL images with Pascal VOC XML annotations (one XML per image). The dataset only contains defective samples; there is no "no defect" class.
- Label order is defined in `src/config/defect_labels.yaml`:
  finger, crack, black_core, thick_line, horizontal_dislocation, short_circuit, vertical_dislocation, star_crack, printing_error, corner, fragment, scratch.
- Images are converted to shape `(1, 224, 224)`; outputs are 12 probabilities (sigmoid) for multi-label prediction.

## Key files
- `pipelines/run_pipeline.py`: orchestrates preprocessing -> training -> evaluation with MLflow logging.
- `pipelines/config/model_config.yaml`: data paths and hyperparameters (CNN depth, batch size, epochs, class weights, early stopping).
- `src/preprocess.py`: builds the dataset/dataloaders, applies transforms, and logs dataset metadata.
- `src/train.py`: ELDefectCNN definition and training loop with macro-recall-driven early stopping; logs the model via `mlflow.pytorch.log_model`.
- `src/evaluate.py`: computes macro and per-class recall and logs metrics to MLflow.
- `src/predict.py`: loads a model from MLflow and runs single-image inference with matching preprocessing.
- `src/promote_model.py`: promotes the latest run to Staging if `val_macro_recall` improves the Production model for `el_defect_classifier`.
- `src/utils/compute_class_weights.py`: helper script to derive tempered class weights from defect counts.
- `docs/training.md`: modeling rationale, imbalance handling, and experiment notes.
- `docs/mltflow_setup.md`: how to configure MLflow tracking and artifact storage.

## Setup
1. Python environment: create/activate a virtualenv.
2. Install dependencies: `pip install -r requirements.txt`.
3. Configure MLflow (tracking URI, artifact store) via `.env` or environment variables; see `docs/mltflow_setup.md`.
4. Place data to match the default config:
   - Images: `data/dataset/JPEGImages/*.jpg`
   - Annotations: `data/dataset/Annotations/*.xml` (filenames match images)

## Run training
```
python pipelines/run_pipeline.py --config pipelines/config/model_config.yaml
```
What happens:
- Loads config and logs it as an artifact.
- Builds train/val DataLoaders with the specified `val_split` and `split_seed`.
- Trains ELDefectCNN for `training.epochs` with AdamW, BCEWithLogitsLoss, and optional class weights (clamped by `max_pos_weight`).
- Logs per-epoch train/val loss and macro recall; evaluation logs per-class recall.
- Saves the trained model artifact to MLflow.

## Configuration highlights (`pipelines/config/model_config.yaml`)
- `data.images_dir` / `data.annotations_dir`: JPEG + XML locations.
- `training.class_weights`: tuned square-root weights for imbalance; adjust as you add data.
- Early stopping: `training.early_stopping.enabled`, `patience`, `min_delta` (uses validation macro recall).
- CNN: `blocks`, `init_channels`, `kernel_size`, `stride`, `padding`.
- Promotion metric: `val_macro_recall` (used across training, evaluation, and promotion).

## Inference
```
from src.predict import load_model, predict_image

model, device = load_model("models:/el_defect_classifier/Staging")
probs = predict_image(model, device, "path/to/image.jpg")
print(probs)  # {label: probability}
```
Probabilities are returned for all classes; apply thresholds per class as needed.

## Model promotion
```
python src/promote_model.py
```
Compares the latest run in the target experiment to the current Production model using `val_macro_recall`. If better, registers the run artifact and transitions it to Staging. Ensure MLflow tracking/registry is configured and `MODEL_NAME` matches `el_defect_classifier` if you customize it.

## Notes and limitations
- The dataset has no defect-free images; the model reports which defects are present, not whether a module is healthy.
- Ultra-rare classes (printing_error, corner, fragment, scratch) may have low recall without more data or class-specific thresholds.
- For modeling decisions, imbalance handling, and future work, see `docs/training.md`.
