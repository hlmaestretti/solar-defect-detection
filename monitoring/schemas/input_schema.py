version: 1

features:
  - name: mean_intensity
    type: float
  - name: std_intensity
    type: float
  - name: p10_intensity
    type: float
  - name: p50_intensity
    type: float
  - name: p90_intensity
    type: float

target:
  name: defect_labels
  type: multi_label
