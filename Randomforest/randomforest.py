# type: ignore
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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

def process_files(all_files):
    """
    Process CSV files to calculate angles and velocities.
    """
    all_data = []
    for file in all_files:
        df = pd.read_csv(file)
        
        # Sort by frame to ensure proper sequence
        df = df.sort_values('frame')
        
        # Combine with original data
        df = pd.concat([df], axis=1)
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

expected_features = (num_landmarks * features_per_landmark)

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



# Save model
joblib.dump(model, 'boxing_technique_classifier_testpkl')