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

1. Change into the Randomforest directory

```bash
cd Randomforest/
```

2. run [randomforest_best.py](randomforest_best.py) to get the model

```bash
python3.12 randomforest_best.py
```

3. run [real_time_detection.py](real_time_detection.py) (camera permission required)

```bash
python3.12 real_time_detection.py
```

Adjustment: In realtime.py, line 132, set it to 0 for webcam, set it to 1 for an external camera (iPhone), or provide a path to a video

Hint:
In the deprecated folder, you’ll find all the older, less accurate Random Forest models along with their corresponding real-time detection scripts.
To run any of these real-time scripts, you first need to generate the models from the Randomforest directory.
Once generated, update the model_path inside each real-time script in deprecated to point to the newly created model file.

```bash
MODEL_PATH = '' # TODO Please add the path to the trained model
```
