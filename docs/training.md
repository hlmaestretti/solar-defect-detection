# Model Training Documentation — EL Defect Classification

### Problem Framing
The goal of this project is to detect defects identification solar module modules based on provided electroluminescence (EL) images.
Each image may contain multiple defect types simultaneously, and the dataset contains only defective samples. This means that this model on indicates what kind of defect is present and does not indicate if there is a defect or not.

Given this, the task is framed as:
- Given a single EL image, predict the probability of each defect type being present.


### Dataset Characteristics & Constraints
Key properties of the dataset that influenced all modeling decisions:
- 12 defect classes
- Highly imbalanced class distribution
    - Some defects appear in thousands of images
    - Others appear in single-digit counts
- Multi-label annotations (images often contain multiple defects)
- Grayscale imagery with high-contrast structural patterns
- Pascal VOC–style XML annotations, used only for label construction

These constraints ruled out:
- softmax / multi-class classification
- accuracy as a primary metric
- naive loss functions without imbalance handling

### Modeling Philosophy
1. Use the data as intended: Don't force the data
Rather than combining this dataset with another dataset and then altering them so that I could have something akin to a binary "Is there a defect", I instead elected to keep this data as it was intended to be used. As such, this model is entirely trained on data that shows defects and does not indicate if a module is defect-free or not.

2. Train from scratch
While I could have found a module that is familiar with EL images or modules, I elected to train from scratch. This is a training exercise and I see training from scratch as more productive towards training than using pretrained models.

3. Optimize for balanced sensitivity, not headline 
Because rare defects are often the most important operationally, macro recall was chosen as:
- the primary evaluation metric
- the early stopping criterion
- the model promotion metric
This ensures improvements benefit all defect types, not just the frequent ones.

### Model Architecture
A custom CNN (ELDefectCNN) was implemented with the following design goals:
- Grayscale image input (1 channel)
- Progressive channel expansion to capture increasingly abstract texture features
- Batch normalization for stable optimization
- Adaptive average pooling to avoid hard-coded spatial dimensions
- Linear output head with 12 logits (one per defect)

The model outputs independent logits per class, allowing multiple defects to be predicted simultaneously.

No softmax is used; probabilities are obtained via sigmoid activation at evaluation or inference time.

### Loss Function & Optimization

#### Base Loss
The baseline loss is: BCEWithLogitsLoss

This is the correct choice for:
- multi-label classification
- independent defect probabilities

#### Optimizer
The model is trained with: AdamW (weight decay enabled)
AdamW was chosen to:
- decouple weight decay from gradient updates
- improve generalization
- stabilize training for a custom CNN trained from scratch

### Handling Class Imbalance
#### Baseline (Unweighted)
The initial unweighted model learned dominant defects well but underperformed on medium-frequency classes, resulting in a validation macro recall of ~0.51.

#### Square-Root Class Weighting

To address imbalance, class weights were introduced using a tempered square-root formulation:
```
w_i = sqrt((N − P_i) / P_i)
```

This:
- increases the influence of underrepresented classes
- avoids the instability caused by extremely large raw weights
- improves macro recall

#### Targeted Weight Adjustment

During experimentation, it was observed that crack defects, while frequent, are visually ambiguous and overlap with other classes (e.g. thick_line, star_crack).

Over-weighting crack caused:
- overly conservative predictions
- reduced crack recall

Reducing the crack weight from ~1.9 → 1.5 restored training stability and improved overall balance, with some macro-recall tradeoff.


### Training Strategy 
#### Train / Validation Split
- A deterministic train/validation split is used
- Splitting is configured via the pipeline config
- Validation data is never used for gradient updates

#### Early Stopping
- Uses validation macro recall to halt training
- Restores weights to best model once training halts

### Evaluation Metrics
For each run, the following metrics are logged:
- Training loss
- Validation loss
- Training macro recall
- Validation macro recall
- Per-class validation recall

Zero-recall for ultra-rare classes (2–7 samples) is expected and that issue is mostly ignored. Class weighting had no impact.

### Final Model Assessment
#### Strengths
- Strong recall on dominant and medium-frequency defects
- Stable training behavior
- No overfitting
- Explainable tradeoffs

#### Known Limitations
- Ultra-rare classes cannot be learned meaningfully with current data volume
- Crack defects remain challenging due to visual ambiguity

Overall, the final weighted model represents a balanced, production-sensible compromise between sensitivity and stability.


### Future Model Improvements
1. Threshold tuning at inference
- Use class-specific thresholds (e.g. lower threshold for crack) to recover recall without destabilizing training
2. Add an additional dataset of defects to better the results for rarer defects. 
3. Incorporate an additional model that identifies whether or not there is a defect in the module that then feeds it to this model if a defect is found.