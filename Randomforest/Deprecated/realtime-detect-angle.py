# type: ignore
import cv2
import mediapipe as mp
import joblib
import numpy as np
from collections import deque
import math

# Configuration
MODEL_PATH = 'boxing_technique_classifier_new_angle_newnew.pkl'
HISTORY_SIZE = 10  # Number of frames to consider for smoothing

# Load trained model
model = joblib.load(MODEL_PATH)

# Define label mapping
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
# Define the limb points (same as in training)
left_arm = [11, 13, 15]  # left_shoulder, left_elbow, left_wrist
right_arm = [12, 14, 16]  # right_shoulder, right_elbow, right_wrist
left_leg = [23, 25, 27]  # left_hip, left_knee, left_ankle
right_leg = [24, 26, 28]  # right_hip, right_knee, right_ankle

# Function to calculate angle between three points in 3D space
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
    
    # Handle cases where vectors are zero length
    if v1_mag * v2_mag == 0:
        return 0
    
    # Calculate angle in radians and then convert to degrees
    # Handle potential numerical errors that could make the ratio outside [-1, 1]
    ratio = dot_product / (v1_mag * v2_mag)
    ratio = max(min(ratio, 1.0), -1.0)
    angle_rad = np.arccos(ratio)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Initialize video capture and smoothing buffer
cap = cv2.VideoCapture("KeyPose_Mediapipe/Ebens winkel/TEST_left_low_kick_6.MOV")
prediction_history = deque(maxlen=HISTORY_SIZE)
print(cv2.__version__)

# Determine video orientation
rotation_angle = 0  # 0 or 180
original_position = cap.get(cv2.CAP_PROP_POS_MSEC)

# Check first 10 frames for orientation detection
for _ in range(10):
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    if results.pose_landmarks:
        # Get relevant landmarks
        nose = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
        left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]

        # Check if nose is below both shoulders (indicating upside-down)
        if nose.y > left_shoulder.y and nose.y > right_shoulder.y:
            rotation_angle = 180
        break

# Reset video to beginning
cap.set(cv2.CAP_PROP_POS_FRAMES, original_position)

prediction_history = deque(maxlen=HISTORY_SIZE)

# For visualization of angles
angle_colors = {
    'left_arm': (255, 0, 0),      # Blue
    'right_arm': (0, 255, 0),     # Green
    'left_leg': (0, 0, 255),      # Red
    'right_leg': (255, 255, 0)    # Cyan
}

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Apply rotation if needed
    if rotation_angle == 180:
        frame = cv2.rotate(frame, cv2.ROTATE_180)

    # Horizontales Spiegeln des Frames
    # frame = cv2.flip(frame, 1)  # 1 für horizontales Spiegeln

    # Convert to RGB and process
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Flip the frame 180 degrees
    # image = cv2.flip(image, 0)  # 0 für vertikales Spiegeln

    results = pose.process(image)

    if results.pose_landmarks:
        # Draw landmarks on the frame
        mp_drawing.draw_landmarks(
            frame, 
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=1)
        )
        
        # Extract keypoints and create feature vector
        landmarks = results.pose_landmarks.landmark
        keypoints = []
        
        # Store 3D coordinates for angle calculation
        point_coords = {}
        
        for i, landmark in enumerate(landmarks):
            # Add to features
            keypoints.extend([
                landmark.x,
                landmark.y,
                landmark.z,
                landmark.visibility
            ])
            
            # Store point coordinates for angle calculation
            point_coords[i] = (landmark.x, landmark.y, landmark.z)
        
        # Calculate angles
        angles = {}
        
        
        if all(i in point_coords for i in left_arm):
            left_arm_angle = calculate_angle(
                point_coords[left_arm[0]], 
                point_coords[left_arm[1]], 
                point_coords[left_arm[2]]
            )
            angles['left_arm_angle'] = left_arm_angle
        else:
            angles['left_arm_angle'] = 0
            
        # Calculate right arm angle
        if all(i in point_coords for i in right_arm):
            right_arm_angle = calculate_angle(
                point_coords[right_arm[0]], 
                point_coords[right_arm[1]], 
                point_coords[right_arm[2]]
            )
            angles['right_arm_angle'] = right_arm_angle
        else:
            angles['right_arm_angle'] = 0
            
        # Calculate left leg angle
        if all(i in point_coords for i in left_leg):
            left_leg_angle = calculate_angle(
                point_coords[left_leg[0]], 
                point_coords[left_leg[1]], 
                point_coords[left_leg[2]]
            )
            angles['left_leg_angle'] = left_leg_angle
        else:
            angles['left_leg_angle'] = 0
            
        # Calculate right leg angle
        if all(i in point_coords for i in right_leg):
            right_leg_angle = calculate_angle(
                point_coords[right_leg[0]], 
                point_coords[right_leg[1]], 
                point_coords[right_leg[2]]
            )
            angles['right_leg_angle'] = right_leg_angle
        else:
            angles['right_leg_angle'] = 0
        
        # Append angles to feature vector
        features = np.array(keypoints + [
            angles['left_arm_angle'], 
            angles['right_arm_angle'], 
            angles['left_leg_angle'], 
            angles['right_leg_angle']
        ]).reshape(1, -1)
        
        # Make prediction with full feature vector including angles
        pred = model.predict(features)[0]
        prediction_history.append(pred)

        # Get most frequent recent prediction
        current_pred = max(set(prediction_history), key=prediction_history.count)
        label = label_map.get(current_pred, "Unknown")

        # Display angle values on frame
        angle_y_pos = 100
        for angle_name, angle_value in angles.items():
            angle_y_pos += 30
            color = angle_colors.get(angle_name, (255, 255, 255))
            cv2.putText(frame, f"{angle_name}: {angle_value:.1f}°", 
                       (20, angle_y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Display result with better visibility
        cv2.rectangle(frame, (10, 10), (400, 60), (0, 0, 0), -1)
        cv2.putText(frame, f"Action: {label}", (20, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Boxing Technique Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()