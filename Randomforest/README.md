# Random Forest Classification

## Features

1. **Kinematic Features**:
   - Joint angles (arms/legs)
   - Angle velocities (deg/frame)
   - Limb endpoint speeds (wrists/ankles)
2. **Temporal Features**:
   - 5-frame rolling averages
   - Velocity peaks
   - Acceleration patterns

## Dataset Requirements

CSV files must contain:

- 132 landmark coordinates (33 points × 4 features)
- Calculated angles/velocities (8 additional features)
- Frame numbers and integer labels

## Output Interpretation

1. **Classification Report**:
   - Precision/recall per technique
   - Macro/micro averages
2. **Feature Importance**:
   - Top 15 influential features
   - Velocity/angle rankings
3. **Visualization**:
   - F1-score comparison chart

## Instruction

1. run [text](randomforest_best.py) to get the model

```bash
python3.12 randomforest_best.py
```

2. run [text](real_time_detection) (camera permission required)

```bash
python3.12 randomforest_best.py
```
