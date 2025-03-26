# type: ignore
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os
import numpy as np
from sklearn.metrics import classification_report

# Define the label mapping
label_map = {
    0: "No Action",
    1: "front_kick_left",
    2: "front_kick_right",
    # 3: "high_kick_left",
    # 4: "high_kick_right", # 
    5: "hook_left_body",
    6: "hook_left_head",
    7: "hook_right_body",
    8: "hook_right_head",
    9: "low_kick_left",
    10: "low_kick_right",
    # 11: "middle_kick_left", # 
    # 12: "middle_kick_right", # 
    13: "roundhouse_kick_left",
    14: "roundhouse_kick_right",
    15: "side_kick_left",
    16: "side_kick_right",
    # 17: "straight_left_body", # 
    18: "straight_left_head", # 
    # 19: "straight_right_body",
    20: "straight_right_head",
}

# Function to extract points from dataframe row
def extract_point(row, point_idx):
    """
    Extract the 3D coordinates of a point from a DataFrame row.
    """
    x = row[f"{point_idx}_x"]
    y = row[f"{point_idx}_y"]
    z = row[f"{point_idx}_z"]
    return np.array([x, y, z])

# Function to calculate velocity between consecutive frames
def calculate_velocity(current_point, previous_point, time_delta=1):
    """
    Calculate the velocity of a point between frames.
    Assumes a constant time delta between frames (can be adjusted if frame rate is known).
    """
    if previous_point is None:
        return 0.0
    
    # Calculate displacement vector
    displacement = current_point - previous_point
    
    # Calculate velocity magnitude (speed)
    speed = np.linalg.norm(displacement) / time_delta
    
    return speed


def process_files(all_files):
    """
    Process CSV files to calculate angles and velocities.
    """
    all_data = []
    for file in all_files:
        df = pd.read_csv(file)
        
        # Sort by frame to ensure proper sequence
        df = df.sort_values('frame')
        
        # Define key points for velocity calculation
        key_points = [
            15,  # Left wrist (for left punches)
            16,  # Right wrist (for right punches)
            27,  # Left ankle (for left kicks)
            28   # Right ankle (for right kicks)
        ]
    
        
        # Add columns for velocity and angular velocity
        velocities = []
        
        # Track previous values for velocity calculation
        prev_points = {point: None for point in key_points}
        
        for idx, row in df.iterrows():
            
            # Calculate velocities for key points
            point_velocities = {}
            for point in key_points:
                current_point = extract_point(row, point)
                velocity = calculate_velocity(current_point, prev_points[point])
                point_velocities[f'point_{point}_velocity'] = velocity
                prev_points[point] = current_point
                
            # Add all velocities to the list
            velocity_data = {
                **point_velocities
            }
            velocities.append(velocity_data)
        
        velocities_df = pd.DataFrame(velocities)
        
        # Combine with original data
        df = pd.concat([df, velocities_df], axis=1)
        all_data.append(df)

    # Combine all processed data
    return pd.concat(all_data, ignore_index=True)


# Load all CSV files
train_data_dir = 'train'
test_data_dir = 'test'
train_files = [os.path.join(train_data_dir, f) for f in os.listdir(train_data_dir) if f.endswith('.csv')]
test_files = [os.path.join(test_data_dir, f) for f in os.listdir(test_data_dir) if f.endswith('.csv')]

# Process the files
train_full_data = process_files(train_files)
test_full_data = process_files(test_files)

print(f"Enhanced data shape: {train_full_data.shape}")
print(f"Number of columns: {len(train_full_data.columns)}")

# Preprocessing for training data
X_train = train_full_data.drop(['frame', 'label'], axis=1)
y_train = train_full_data['label']

# Preprocessing for test data
X_test_full = test_full_data.drop(['frame', 'label'], axis=1)
y_test = test_full_data['label']

# Calculate expected features
num_landmarks = 33
features_per_landmark = 4  # x, y, z, visibility
key_point_velocities = 4  # velocities for key points (wrists, ankles)

expected_features = (num_landmarks * features_per_landmark) + key_point_velocities

assert X_train.shape[1] == expected_features, \
    f"Expected {expected_features} features, got {X_train.shape[1]}"


# Train Random Forest
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    class_weight='balanced',
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate the model on test data
y_pred = model.predict(X_test_full)

print()

# Print evaluation metrics
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=[label_map[i] for i in sorted(label_map.keys())]))

# Feature importance analysis
feature_names = list(X_train.columns)
feature_importance = model.feature_importances_
sorted_idx = np.argsort(feature_importance)[::-1]

print("\nTop 15 Important Features:")
for i in range(15):
    idx = sorted_idx[i]
    print(f"{feature_names[idx]}: {feature_importance[idx]:.4f}")


# Check if velocity/angle features are among top features
velocity_features = ['point_15_velocity', 'point_16_velocity', 'point_27_velocity', 'point_28_velocity', ]

angle_features = ['left_arm_angle', 'right_arm_angle', 'left_leg_angle', 'right_leg_angle']

print("\nRanking of Velocity and Angle Features:")
for feature in velocity_features + angle_features:
    try:
        rank = list(np.argsort(feature_importance)[::-1]).index(feature_names.index(feature))
        print(f"{feature}: Rank {rank+1} (importance: {feature_importance[feature_names.index(feature)]:.4f})")
    except ValueError:
        print(f"{feature}: Not found in feature list")

# Save model
joblib.dump(model, 'boxing_technique_classifier_velocity.pkl')