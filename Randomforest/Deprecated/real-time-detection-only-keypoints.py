# type: ignore
import cv2
import mediapipe as mp
import joblib
import numpy as np
from collections import deque

# Configuration
MODEL_PATH = 'boxing_technique_classifier_angle_with_velocity.pkl' 
HISTORY_SIZE = 10  # Number of frames to consider for smoothing

# Load trained model
model = joblib.load(MODEL_PATH)

# Define label mapping
label_map = {
    0: "No Action",
    1: "front_kick_left",
    2: "front_kick_right",
    # 3: "high_kick_left",
    # 4: "high_kick_right",
    5: "hook_left_body",
    6: "hook_left_head",
    7: "hook_right_body",
    8: "hook_right_head",
    9: "low_kick_left",
    10: "low_kick_right",
    # 11: "middle_kick_left",
    # 12: "middle_kick_right",
    13: "roundhouse_kick_left",
    14: "roundhouse_kick_right",
    15: "side_kick_left",
    16: "side_kick_right",
    # 17: "straight_left_body",
    18: "straight_left_head",
    # 19: "straight_right_body",
    20: "straight_right_head",
}

# Define the limb points (same as in training)
left_arm = [11, 13, 15]  # left_shoulder, left_elbow, left_wrist
right_arm = [12, 14, 16]  # right_shoulder, right_elbow, right_wrist
left_leg = [23, 25, 27]  # left_hip, left_knee, left_ankle
right_leg = [24, 26, 28]  # right_hip, right_knee, right_ankle

# Define key points for velocity calculation
key_points = [
    15,  # Left wrist (for left punches)
    16,  # Right wrist (for right punches)
    27,  # Left ankle (for left kicks)
    28   # Right ankle (for right kicks)
]

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

# Function to calculate velocity between consecutive frames
def calculate_velocity(current_point, previous_point, time_delta=1):
    """
    Calculate the velocity of a point between frames.
    Assumes a constant time delta between frames.
    """
    if previous_point is None:
        return 0.0
    
    # Calculate displacement vector
    displacement = np.array(current_point) - np.array(previous_point)
    
    # Calculate velocity magnitude (speed)
    speed = np.linalg.norm(displacement) / time_delta
    
    return speed

# Function to calculate angular velocity
def calculate_angular_velocity(current_angle, previous_angle, time_delta=1):
    """
    Calculate the angular velocity between frames.
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
    
    # Calculate angular velocity (absolute value for magnitude)
    angular_velocity = abs(angle_change) / time_delta
    
    return angular_velocity

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Initialize video capture and smoothing buffer
cap = cv2.VideoCapture("KeyPose_Mediapipe/Ebens winkel/DONE_side_kick_left_8.MOV") # Set to 0 for webcam, Set to 1 for external camera (IPhone)

fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("RandomForest_Video_Annotated.mp4", fourcc, fps, (width, height))

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


# For visualization of angles and velocities
angle_colors = {
    'left_arm': (255, 0, 0),      # Blue
    'right_arm': (0, 255, 0),     # Green
    'left_leg': (0, 0, 255),      # Red
    'right_leg': (255, 255, 0)    # Cyan
}

velocity_colors = {
    'point_15_velocity': (255, 0, 255),    # Magenta
    'point_16_velocity': (255, 165, 0),    # Orange
    'point_27_velocity': (0, 255, 255),    # Yellow
    'point_28_velocity': (128, 0, 128)     # Purple
}

# Store previous values for velocity calculations
prev_angles = {
    'left_arm': None,
    'right_arm': None,
    'left_leg': None,
    'right_leg': None
}

prev_points = {point: None for point in key_points}
prev_center = None

frame_counter = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        break  # Changed from continue to break to avoid infinite loop if video ends

    # Apply rotation if needed
    if rotation_angle == 180:
        frame = cv2.rotate(frame, cv2.ROTATE_180)

    frame_counter += 1

    # Convert to RGB and process
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = pose.process(image)

    if results.pose_landmarks:
        # Setze den Hintergrund auf Weiß
        frame[:] = (255, 255, 255)  # Weißer Hintergrund

        # Zeichne nur die Schlüsselstellen auf dem weißen Hintergrund
        mp_drawing.draw_landmarks(
            frame, 
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=4, circle_radius=3),  # Dickere Linien und größere Kreise
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=4, circle_radius=2)   # Dickere Linien und größere Kreise
        )
        
        # Extract keypoints and create feature vector
        landmarks = results.pose_landmarks.landmark
        keypoints = []
        
        # Store 3D coordinates for angle and velocity calculations
        point_coords = {}
        
        for i, landmark in enumerate(landmarks):
            # Add to features
            keypoints.extend([
                landmark.x,
                landmark.y,
                landmark.z,
                landmark.visibility
            ])
            
            # Store point coordinates
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
        
        # Calculate angular velocities
        angular_velocities = {}
        angular_velocities['left_arm_angular_velocity'] = calculate_angular_velocity(
            angles['left_arm_angle'], prev_angles['left_arm'])
        angular_velocities['right_arm_angular_velocity'] = calculate_angular_velocity(
            angles['right_arm_angle'], prev_angles['right_arm'])
        angular_velocities['left_leg_angular_velocity'] = calculate_angular_velocity(
            angles['left_leg_angle'], prev_angles['left_leg'])
        angular_velocities['right_leg_angular_velocity'] = calculate_angular_velocity(
            angles['right_leg_angle'], prev_angles['right_leg'])
        
        # Update previous angles
        prev_angles['left_arm'] = angles['left_arm_angle']
        prev_angles['right_arm'] = angles['right_arm_angle']
        prev_angles['left_leg'] = angles['left_leg_angle']
        prev_angles['right_leg'] = angles['right_leg_angle']
        
        # Calculate velocities for key points
        velocities = {}
        for point in key_points:
            if point in point_coords:
                current_point = point_coords[point]
                velocity = calculate_velocity(current_point, prev_points[point])
                velocities[f'point_{point}_velocity'] = velocity
                prev_points[point] = current_point
            else:
                velocities[f'point_{point}_velocity'] = 0
        
        # Construct the complete feature vector
        features = np.array(keypoints + [
            angles['left_arm_angle'], 
            angles['right_arm_angle'], 
            angles['left_leg_angle'], 
            angles['right_leg_angle'],
            angular_velocities['left_arm_angular_velocity'],
            angular_velocities['right_arm_angular_velocity'],
            angular_velocities['left_leg_angular_velocity'],
            angular_velocities['right_leg_angular_velocity'],
            velocities['point_15_velocity'],
            velocities['point_16_velocity'], 
            velocities['point_27_velocity'], 
            velocities['point_28_velocity'],
        ]).reshape(1, -1)
        
        # Make prediction with full feature vector
        try:
            pred = model.predict(features)[0]
            prediction_history.append(pred)

            # Get most frequent recent prediction
            current_pred = max(set(prediction_history), key=prediction_history.count)
            label = label_map.get(current_pred, "Unknown")
        except Exception as e:
            print(f"Prediction error: {e}")
            print(f"Feature vector shape: {features.shape}")
            label = "Error in prediction"
        
        
        # Display result with better visibility
        cv2.rectangle(frame, (10, 10), (400, 100), (0, 0, 0), -1)  # Größerer Hintergrund für bessere Sichtbarkeit
        cv2.putText(frame, f"Action: {label}", (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)  # Weißer Text für bessere Lesbarkeit
        cv2.putText(frame, f"Frame: {frame_counter}", 
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)  # Hellgrauer Text für Frame-Nummer

    # Display the frame
    cv2.imshow('Boxing Technique Detection', frame)

    # Save frame to a video file
    out.write(frame)
    

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()