# type: ignore
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os
import numpy as np
from sklearn.metrics import classification_report
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

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

# Function to calculate the angle between three points in 3D space
def calculate_angle(point1, point2, point3):
    """
    Calculate the angle between three points in 3D space.
    point2 is the vertex of the angle.
    """
    # Vector from point2 to point1
    v1 = np.array([point1[0] - point2[0], point1[1] - point2[1], point1[2] - point2[2]])
    # Vector from point2 to point3
    v2 = np.array([point3[0] - point2[0], point3[1] - point2[1], point3[2] - point2[2]])
    
    # Calculate dot product
    dot_product = np.dot(v1, v2)
    
    # Calculate magnitudes
    v1_mag = np.linalg.norm(v1)
    v2_mag = np.linalg.norm(v2)
    
    # Calculate angle in radians and then convert to degrees
    # Handle potential numerical errors that could make the ratio outside [-1, 1]
    ratio = dot_product / (v1_mag * v2_mag)
    ratio = max(min(ratio, 1.0), -1.0)
    angle_rad = np.arccos(ratio)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg

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

# Function to calculate angle velocity
def calculate_angle_velocity(current_angle, previous_angle, time_delta=1):
    """
    Calculate the angle velocity between frames.
    """
    if previous_angle is None:
        return 0.0
    
    # Calculate change in angle
    angle_change = current_angle - previous_angle
    
    # Handle angle wrapping (e.g., if angle goes from 350 to 10 degrees)
    if angle_change > 180:
        angle_change -= 360
    elif angle_change < -180:
        angle_change += 360
    
    # Calculate angle velocity
    angle_velocity = abs(angle_change) / time_delta
    
    return angle_velocity


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
        
        # Define the limb points for angle calculation
        left_arm = [11, 13, 15]  # shoulder, elbow, wrist
        right_arm = [12, 14, 16]
        left_leg = [23, 25, 27]  # hip, knee, ankle
        right_leg = [24, 26, 28]
        
        # Add columns for velocity and angle velocity
        velocities = []
        angles = []
        
        # Track previous values for velocity calculation
        prev_points = {point: None for point in key_points}
        prev_angles = {'left_arm': None, 'right_arm': None, 'left_leg': None, 'right_leg': None}
        
        for idx, row in df.iterrows():
            # Calculate angles
            la_point1 = extract_point(row, left_arm[0])
            la_point2 = extract_point(row, left_arm[1])
            la_point3 = extract_point(row, left_arm[2])
            
            ra_point1 = extract_point(row, right_arm[0])
            ra_point2 = extract_point(row, right_arm[1])
            ra_point3 = extract_point(row, right_arm[2])
            
            ll_point1 = extract_point(row, left_leg[0])
            ll_point2 = extract_point(row, left_leg[1])
            ll_point3 = extract_point(row, left_leg[2])
            
            rl_point1 = extract_point(row, right_leg[0])
            rl_point2 = extract_point(row, right_leg[1])
            rl_point3 = extract_point(row, right_leg[2])
            
            # Calculate current angles
            left_arm_angle = calculate_angle(la_point1, la_point2, la_point3)
            right_arm_angle = calculate_angle(ra_point1, ra_point2, ra_point3)
            left_leg_angle = calculate_angle(ll_point1, ll_point2, ll_point3)
            right_leg_angle = calculate_angle(rl_point1, rl_point2, rl_point3)
            
            # Calculate angle velocities
            left_arm_av = calculate_angle_velocity(left_arm_angle, prev_angles['left_arm'])
            right_arm_av = calculate_angle_velocity(right_arm_angle, prev_angles['right_arm'])
            left_leg_av = calculate_angle_velocity(left_leg_angle, prev_angles['left_leg'])
            right_leg_av = calculate_angle_velocity(right_leg_angle, prev_angles['right_leg'])
            
            # Update previous angles
            prev_angles['left_arm'] = left_arm_angle
            prev_angles['right_arm'] = right_arm_angle
            prev_angles['left_leg'] = left_leg_angle
            prev_angles['right_leg'] = right_leg_angle
            
            # Add angles to list
            angles.append({
                'left_arm_angle': left_arm_angle,
                'right_arm_angle': right_arm_angle,
                'left_leg_angle': left_leg_angle,
                'right_leg_angle': right_leg_angle,
                'left_arm_angle_velocity': left_arm_av,
                'right_arm_angle_velocity': right_arm_av,
                'left_leg_angle_velocity': left_leg_av,
                'right_leg_angle_velocity': right_leg_av
            })
            
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
        
        # Convert to DataFrame and add to original data
        angles_df = pd.DataFrame(angles)
        velocities_df = pd.DataFrame(velocities)
        
        # Combine with original data
        df = pd.concat([df, angles_df, velocities_df], axis=1)
        all_data.append(df)

    # Combine all processed data
    return pd.concat(all_data, ignore_index=True)


# Load all CSV files
train_data_dir = 'data/train'
test_data_dir = 'data/test'
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
angle_features = 4  # left/right arm/leg angles
angle_velocity_features = 4  # angle velocities for each angle
key_point_velocities = 4  # velocities for key points (wrists, ankles)

expected_features = (num_landmarks * features_per_landmark) + angle_features + angle_velocity_features + key_point_velocities

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

# Plot the first five trees in the forest
# for i in range(5):
#     plt.figure(figsize=(20,10))
#     plot_tree(model.estimators_[i], filled=True, feature_names=X_train.columns, class_names=[label_map[i] for i in sorted(label_map.keys())])
#     plt.title(f"Decision Tree {i+1} from Random Forest")
#     plt.show()

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
velocity_features = ['point_15_velocity', 'point_16_velocity', 'point_27_velocity', 'point_28_velocity', 'left_arm_angle_velocity', 'right_arm_angle_velocity', 
                    'left_leg_angle_velocity', 'right_leg_angle_velocity']

angle_features = ['left_arm_angle', 'right_arm_angle', 'left_leg_angle', 'right_leg_angle']

print("\nRanking of Velocity and Angle Features:")
for feature in velocity_features + angle_features:
    try:
        rank = list(np.argsort(feature_importance)[::-1]).index(feature_names.index(feature))
        print(f"{feature}: Rank {rank+1} (importance: {feature_importance[feature_names.index(feature)]:.4f})")
    except ValueError:
        print(f"{feature}: Not found in feature list")

# Save model
joblib.dump(model, 'randomforest_best_model.pkl')


# Classification Report auswerten
report = classification_report(y_test, y_pred, target_names=[label_map[i] for i in sorted(label_map.keys())], output_dict=True)

# F1-Scores für jede Klasse extrahieren
f1_scores = {label: report[label]['f1-score'] for label in label_map.values()}

# Sortieren der F1-Scores für eine bessere Darstellung
sorted_f1_scores = dict(sorted(f1_scores.items(), key=lambda item: item[1], reverse=True))

# Erstellen des Balkendiagramms
plt.figure(figsize=(12, 8))
plt.barh(list(sorted_f1_scores.keys()), list(sorted_f1_scores.values()), color='skyblue')
plt.xlabel('F1-Score')
plt.title('F1-Score of each technique')
plt.gca().invert_yaxis()  # Sortierung von oben nach unten
plt.xlim(0, 1)  # F1-Score liegt zwischen 0 und 1
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Beschriftungen hinzufügen
for i, (label, score) in enumerate(sorted_f1_scores.items()):
    plt.text(score, i, f'{score:.2f}', va='center', ha='left')

plt.show()