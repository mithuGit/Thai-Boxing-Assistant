import cv2
import mediapipe as mp
import numpy as np
import csv
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

# Video Processing Functions

def get_video_file():
    """Prompt user for video file path through terminal input."""
    video_path = input("Please enter the path to the video (e.g. video.mp4): ").strip()
    return video_path

def process_video(video_path):
    """
    Process video to extract pose landmarks using MediaPipe.
    Performs automatic orientation detection and normalization relative to hip position.
    Returns sequence of feature vectors (33 landmarks with x,y,z,visibility per frame).
    """
    cap = cv2.VideoCapture(video_path)
    mp_pose = mp.solutions.pose
    
    # Automatic orientation detection
    rotate_video = False
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose_detector:
        found_frame = False
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose_detector.process(image_rgb)
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                nose_y = landmarks[0].y
                hip1_y = landmarks[23].y
                hip2_y = landmarks[24].y
                hips_center_y = (hip1_y + hip2_y) / 2.0
                # Detect upside-down orientation
                if nose_y > hips_center_y:
                    rotate_video = True
                    print("Video appears to be upside down. Rotate it 180° before processing.")
                else:
                    print("Video orientation is correct.")
                found_frame = True
                break
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Reset video stream
    
    # Main processing loop
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
    sequence = []
    
    frame_index = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Apply rotation if needed
        if rotate_video:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            features = []
            for lm in landmarks:
                features.extend([lm.x, lm.y, lm.z, lm.visibility])
            sequence.append(features)
        else:
            sequence.append([0]*132) # Fallback for missing landmarks
        
        frame_index += 1
    
    cap.release()
    sequence = np.array(sequence)
    print(f"Processed {len(sequence)} Frames.")
    return sequence

# REFERENCE DATA HANDLING

def load_reference_sequence(csv_file, normalize=True):
    """
    Load reference sequence from CSV file.
    Expected format: 132 values per row (33 landmarks * 4 features).
    """
    sequence = []
    expected_length = 132
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)  # Skip header
        for row in reader:
            if not row:
                continue
            # Extract features (excluding frame number and label)
            features = np.array(list(map(float, row[1:-1])))
            if len(features) != expected_length:
                print(f"Skipped because length {len(features)} does not fit.")
                continue
            sequence.append(features)
    sequence = np.array(sequence)
    print(f"Reference sequence loaded from '{csv_file}' with shape: {sequence.shape}")
    return sequence

# SEQUENCE PROCESSING

def segment_sequence(sequence, window_size=35, step=5):
    """
    Split sequence into overlapping windows.
    Returns list of tuples (start_frame, end_frame, segment_data).
    """
    segments = []
    num_frames = sequence.shape[0]
    for start in range(0, num_frames - window_size + 1, step):
        end = start + window_size
        segment = sequence[start:end]
        segments.append((start, end, segment))
    print(f"Creates {len(segments)} segments.")
    return segments

def compute_dtw_distance(seq1, seq2):
    """Calculate Dynamic Time Warping distance between two sequences."""
    distance, path = fastdtw(seq1, seq2, dist=euclidean)
    return distance, path

# PUNCH DETECTION LOGIC
    
def detect_punches_multiple(segments, ref_sequences):
    """
    Detect punches by comparing segments to multiple reference sequences.
    Uses type-specific thresholds for minimum DTW distance.
    Returns detected punches and their distances.
    """
    thresholds = {
        "Hook body Left": 18, #sliding window problem: overlapping windows, no pause between punches 18
        "Hook head Left": 20, #sliding window + missing keypoints on left side 20
        "Hook head Right": 21, #21
        "Hook body Right": 21, #21
        "Front Kick Right": 25, #25
        "Front Kick Left": 37, #37
        "Straight Head Left": 20, #20
        "Straight Head Right": 18, #18
        "Low Kick Left": 24, #24
        "Low Kick Right": 28, #28
        "Roundhouse Kick Left": 30, #30
        "Roundhouse Kick Right": 30, #30
        "Side Kick Left": 45, #45
        "Side Kick Right": 40  #40
        }

    results = []
    best_distances = []

    for start, end, segment in segments:
        min_distance = float('inf')
        best_punch = None

        # Find best matching reference sequence
        for punch_type, ref_seq in ref_sequences.items():
            distance, _ = compute_dtw_distance(segment, ref_seq)
            
            if distance < min_distance:
                min_distance = distance
                best_punch = punch_type

        # Check against type-specific threshold
        if best_punch and min_distance < thresholds.get(best_punch, float('inf')):
            results.append((start, end, min_distance, best_punch))
            best_distances.append(min_distance)

    print("Recognized punches (Segments):", results)
    return results, best_distances

# RESULT HANDLING

def annotate_video(video_path, output_path, results, flip_video=False):
    """
    Annotate video with detection results.
    Handles automatic orientation correction and optional horizontal flip.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Video could not be opened.")
        return

    # Automatic orientation detection
    rotate_video = False
    mp_pose = mp.solutions.pose
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose_detector:
        found_frame = False
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results_pose = pose_detector.process(image_rgb)
            if results_pose.pose_landmarks:
                landmarks = results_pose.pose_landmarks.landmark
                # Calculate vertical positions for orientation detection
                nose_y = landmarks[0].y
                hip1_y = landmarks[23].y
                hip2_y = landmarks[24].y
                hips_center_y = (hip1_y + hip2_y) / 2.0
                if nose_y > hips_center_y:
                    rotate_video = True
                    print("Video appears to be upside down. Rotate the output video 180°.")
                else:
                    print("Video orientation seems correct for the output video.")
                found_frame = True
                break
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Video writer setup    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Frame processing loop
    frame_counter = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Apply transformations
        if rotate_video:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        
        if flip_video:
            frame = cv2.flip(frame, 1)
        
        # Add annotations for detected punches
        for start, end, dist, punch_type in results:
            if start <= frame_counter < end:
                top_left = (50, 50)
                bottom_right = (width - 50, height - 50)
                cv2.rectangle(frame, top_left, bottom_right, (0, 0, 255), 3)
                elapsed_time = (frame_counter - start) / fps
                cv2.putText(frame, f"{punch_type}! ({dist:.2f})", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv2.LINE_AA)
                cv2.putText(frame, f"Time: {elapsed_time:.1f}s", (50, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, f"Frame: {frame_counter}", (50, 300),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                break
        
        out.write(frame)
        frame_counter += 1
    
    cap.release()
    out.release()
    print(f"Annotated video was saved as '{output_path}'.")

def main():
    # Get video path from user
    video_file = get_video_file()
    if not video_file:
        print("No video specified.")
        return
    print("Selected video:", video_file)
    
    # Process video to get pose sequence
    sequence = process_video(video_file)
    
    # Load reference sequences
    ref_csv_files = {
        "Hook body Left": "DTW/references/hook_body_left.csv",
        "Hook head Left": "DTW/references/hook_left_head.csv",
        "Hook head Right": "DTW/references/hook_head_right.csv",
        "Hook body Right": "DTW/references/hook_body_right.csv",
        "Front Kick Right": "DTW/references/frontkick_right.csv",
        "Front Kick Left": "DTW/references/frontkick_left.csv",
        "Straight Head Left": "DTW/references/straight_left.csv",
        "Straight Head Right": "DTW/references/straight_right.csv",
        "Low Kick Left": "DTW/references/lowkick_left.csv",
        "Low Kick Right": "DTW/references/lowkick_right.csv",
        "Roundhouse Kick Left": "DTW/references/roundhouse_left.csv",
        "Roundhouse Kick Right": "DTW/references/roundhouse_right.csv",
        "Side Kick Left": "DTW/references/sidekick_left.csv",
        "Side Kick Right": "DTW/references/sidekick_right.csv",
    }
    ref_sequences = {}

    for punch_type, csv_file in ref_csv_files.items():
        print(f"Load reference sequence for '{punch_type}' from '{csv_file}'")
        ref_sequences[punch_type] = load_reference_sequence(csv_file)
    
    # Detect punches in video segments
    segments = segment_sequence(sequence, window_size=35, step=35)
    results, distances = detect_punches_multiple(segments, ref_sequences)

    # Save and visualize results
    output_video = "annotated_output.mp4"
    annotate_video(video_file, output_video, results)

if __name__ == "__main__":
    main()