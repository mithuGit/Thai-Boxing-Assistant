import cv2
import os
import mediapipe as mp
import csv

# Initialize Mediapipe pose detection components
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def process_videos(video_path):
    results_dir = "data/test_label"  # TODO Add path to save results
    results_dir_video = "data/test_label_videos"  # TODO Add path to save results
    os.makedirs(results_dir, exist_ok=True)  # Create the folder if it does not exist
    os.makedirs(results_dir_video, exist_ok=True)  # Create the folder if it does not exist

    print(f"Processing video: {video_path}")
    csv_filename = os.path.join(results_dir, f"{os.path.splitext(os.path.basename(video_path))[0]}.csv")
    output_video_path = os.path.join(results_dir_video, f"{os.path.splitext(os.path.basename(video_path))[0]}_keypoints.mp4")

    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Initialize VideoWriter
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

    with open(csv_filename, mode="w", newline="") as file:
        csv_writer = csv.writer(file)
        
        # Write the header row (landmark names)
        landmark_names = [f"{name}_{coord}" for name in range(33) for coord in ["x", "y", "z", "visibility"]]
        csv_writer.writerow(["frame"] + landmark_names + ["label"])

        with mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
        ) as pose:
            
            frame_idx = 0
            label = 0  # Initialize punch label
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    print("End of video or error reading frame.")
                    break
                
                # Rotate the frame if necessary # TODO
                frame = cv2.rotate(frame, cv2.ROTATE_180)

                # Check for key press to toggle label # TODO
                if cv2.waitKey(100) & 0xFF == ord('p'):  # Press 'p' to toggle punch label
                    label = 17
                    print("Label set to 17")
                else:
                    label = 0

                # Convert the BGR image to RGB before processing
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process the image and detect pose
                results = pose.process(frame_rgb)
                
                # Extract keypoints and store them in CSV
                if results.pose_landmarks:
                    keypoints = []
                    for landmark in results.pose_landmarks.landmark:
                        keypoints.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
                    csv_writer.writerow([frame_idx] + keypoints + [label])

                    # Draw pose annotations
                    mp_drawing.draw_landmarks(
                        frame,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
                    )
                
                # Show processed video
                cv2.imshow("Processed Video", frame)

                # Write the frame with keypoints to the output video
                out.write(frame)
                
                # Press 'q' to exit early
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                frame_idx += 1  # Increment frame index

        # Release resources
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"Pose keypoints saved to {csv_filename}")
        print(f"Processed video saved to {output_video_path}")

if __name__ == "__main__":
    video_path = 'DTW/test_videos/TEST_right_hook_body_6.MOV' # TODO Add path to video you want to process
    process_videos(video_path)