# type: ignore
import cv2
import mediapipe as mp
import joblib
import numpy as np
from collections import deque

# Configuration
MODEL_PATH = 'boxing_technique_classifier_new.pkl'
HISTORY_SIZE = 10  # Number of frames to consider for smoothing

# Load trained model
model = joblib.load(MODEL_PATH)

# Define label mapping (update according to your classes)
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

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils  # Hinzugefügt für das Zeichnen der Keypoints

# Initialize video capture and smoothing buffer
cap = cv2.VideoCapture("KeyPose_Mediapipe/Ebens winkel/TEST_straight_right_6.MOV")
print(cv2.__version__)
prediction_history = deque(maxlen=HISTORY_SIZE)

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
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    # Apply rotation if needed
    if rotation_angle == 180:
        frame = cv2.rotate(frame, cv2.ROTATE_180)

    # Horizontales Spiegeln des Frames
    # frame = cv2.flip(frame, 1)  # 1 für horizontales Spiegeln

    # Convert to RGB and process
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # flip the frame to 180 degree
    # image = cv2.flip(image, 0)  # 0 für vertikales Spiegeln

    results = pose.process(image)

    if results.pose_landmarks:
        # Zeichne die Keypoints auf dem Frame
        mp_drawing.draw_landmarks(
            frame, 
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=1)
        )
        
        # Extract keypoints in order (33 landmarks, 4 features each)
        keypoints = []
        for landmark in results.pose_landmarks.landmark:
            keypoints.extend([
                landmark.x,
                landmark.y,
                landmark.z,
                landmark.visibility
            ])
        
        # Make prediction
        features = np.array(keypoints).reshape(1, -1)
        pred = model.predict(features)[0]
        prediction_history.append(pred)

        # Get most frequent recent prediction
        current_pred = max(set(prediction_history), key=prediction_history.count)
        label = label_map.get(current_pred, "Unknown")

        # Display result with better visibility
        cv2.rectangle(frame, (10, 10), (400, 60), (0, 0, 0), -1)
        cv2.putText(frame, f"Aktion: {label}", (20, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Hier wird das gesamte Frame angezeigt, das bereits gespiegelt wurde
    cv2.imshow('Boxing Technique Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() 