
import cv2
import mediapipe as mp
import numpy as np
import math
import os
import pandas as pd
import streamlit as st
from PIL import Image

# Initialize MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Function to calculate the angle between three points
def calculate_angle(point1, point2, point3):
    vec1 = [point1[0] - point2[0], point1[1] - point2[1]]
    vec2 = [point3[0] - point2[0], point3[1] - point2[1]]
    dot_product = vec1[0] * vec2[0] + vec1[1] * vec2[1]
    mag1 = math.sqrt(vec1[0]**2 + vec1[1]**2)
    mag2 = math.sqrt(vec2[0]**2 + vec2[1]**2)
    radians = math.acos(dot_product / (mag1 * mag2))
    return math.degrees(radians)

# Function to classify orientation based on the position of the nose and shoulders
def classify_orientation(nose, left_shoulder, right_shoulder):
    shoulder_midpoint = [(left_shoulder[0] + right_shoulder[0]) / 2, (left_shoulder[1] + right_shoulder[1]) / 2]
    dx = nose[0] - shoulder_midpoint[0]
    dy = nose[1] - shoulder_midpoint[1]
    angle = np.arctan2(dy, dx) * 180.0 / np.pi

    if 70 <= abs(angle) <= 110:
        return "Front"
    elif nose[0] < left_shoulder[0]:
        return "Left"
    elif nose[0] > right_shoulder[0]:
        return "Right"
    else:
        return "Uncertain"

# Function to calculate angles for the specified joint sets
def calculate_joint_angles(landmarks):
    angles = {}
    angles['15_13_11'] = calculate_angle([landmarks[15].x, landmarks[15].y], 
                                         [landmarks[13].x, landmarks[13].y], 
                                         [landmarks[11].x, landmarks[11].y])
    angles['12_14_16'] = calculate_angle([landmarks[12].x, landmarks[12].y], 
                                         [landmarks[14].x, landmarks[14].y], 
                                         [landmarks[16].x, landmarks[16].y])
    return angles

# Function to load angles from CSV for different facing directions
def load_angles_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    angles_dict = {}
    for _, row in df.iterrows():
        angles_dict[row['Landmark Pair']] = {
            'Front': row['Front (degrees)'],
            'Right': row['Right (degrees)'],
            'Left': row['Left (degrees)']
        }
    return angles_dict

# Function to detect pose and validate angles based on facing direction.
def detect_pose(results, ideal_angles, direction):
    feedback = []
    try:
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            correct = [1] * len(landmarks)

            joint_angles = calculate_joint_angles(landmarks)

            print("Calculated Angles from Video Frame:")
            for key, angle in joint_angles.items():
                print(f"{key}: {angle:.2f}")

            angle_threshold = 20

            # Check angles against thresholds based on direction.
            for key, angle in joint_angles.items():
                ideal_angle = ideal_angles[key][direction]
                if not (ideal_angle - angle_threshold <= angle <= ideal_angle + angle_threshold):
                    points = list(map(int, key.split('_')))
                    middle_point = points[1]  # The middle point is always the second in our angle keys
                    correct[middle_point] = 0
                    feedback.append(f"Angle at {key} should be between {ideal_angle - angle_threshold:.2f} and {ideal_angle + angle_threshold:.2f} degrees, but is {angle:.2f}.")

            accuracy = sum(correct) / len(correct) * 100
            pose_name = "Correct" if accuracy == 100 else "Incorrect"

            feedback_str = ' '.join(feedback) if feedback else "Pose is correct"
            return accuracy, pose_name, correct, feedback_str
        else:
            print("No pose landmarks detected in this frame.")
            return 0.0, "No pose detected", [], "No pose detected"

    except Exception as e:
        print(f"Error during pose detection: {e}")
        return 0.0, "Error", [], f"Error: {str(e)}"

# Function to draw landmarks and connections on the image.
def draw_landmarks(image, landmarks, correct, direction, accuracy, feedback_str):
    h, w, _ = image.shape
    for idx, landmark in enumerate(landmarks.landmark):
        color = (0, 255, 0) if correct[idx] == 1 else (0, 0, 255)  # Green for correct, Red for incorrect
        cx, cy = int(landmark.x * w), int(landmark.y * h)
        cv2.circle(image, (cx, cy), 5, color, -1)

    # Draw connections
    for connection in mp_pose.POSE_CONNECTIONS:
        start_idx, end_idx = connection
        start_point = landmarks.landmark[start_idx]
        end_point = landmarks.landmark[end_idx]
        color = (0, 255, 0) if correct[start_idx] == 1 and correct[end_idx] == 1 else (0, 0, 255)
        start = (int(start_point.x * w), int(start_point.y * h))
        end = (int(end_point.x * w), int(end_point.y * h))
        cv2.line(image, start, end, color, 2)

    # Overlay the direction, accuracy, and feedback on the image
    cv2.putText(image, f'Facing: {direction}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(image, f'Accuracy: {accuracy:.2f}%', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(image, f'Feedback: {feedback_str}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

# Function to process video or webcam feed with real-time visualization and feedback
def process_video(video_source, ideal_angles, is_webcam=False):
    cap = cv2.VideoCapture(video_source)
    frame_placeholder = st.empty()  # To hold the frames dynamically

    correct_poses = 0
    total_frames = 0

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x, landmarks[mp_pose.PoseLandmark.NOSE.value].y]
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]

                h, w, _ = frame.shape
                nose = [int(nose[0] * w), int(nose[1] * h)]
                left_shoulder = [int(left_shoulder[0] * w), int(left_shoulder[1] * h)]
                right_shoulder = [int(right_shoulder[0] * w), int(right_shoulder[1] * h)]

                direction = classify_orientation(nose, left_shoulder, right_shoulder)

                accuracy, pose_name, correct, feedback_str = detect_pose(results, ideal_angles, direction)

                # Draw landmarks with feedback, direction, and accuracy
                draw_landmarks(frame, results.pose_landmarks, correct, direction, accuracy, feedback_str)

                # Display frame in Streamlit
                frame_placeholder.image(frame, channels="BGR")

                # Update counters
                if accuracy == 100:
                    correct_poses += 1

                total_frames += 1

    cap.release()

    if not is_webcam:
        st.write(f"Total frames processed: {total_frames}")
        st.write(f"Correct poses: {correct_poses}")
        st.write(f"Accuracy: {(correct_poses / total_frames) * 100:.2f}%")
    else:
        st.write(f"Real-time pose feedback: Correct: {correct_poses} / {total_frames} frames.")

# --- Streamlit Web App Interface ---

st.title("Pose Detection Web App")

# Step 1: CSV File Selection (For Ideal Angles)
csv_folder = "CSV's"
csv_files = [f for f in os.listdir(csv_folder) if f.endswith('.csv')]
selected_csv = st.selectbox("Choose a CSV file:", csv_files)

if selected_csv:
    csv_path = os.path.join(csv_folder, selected_csv)
    ideal_angles = load_angles_from_csv(csv_path)
    st.write(f"**You selected:** {selected_csv}")
    st.write("Ideal angles loaded successfully.")

    # Step 2: Choose Input Method
    input_method = st.radio("Choose input method:", ["Upload Video", "Use Webcam"])

    if input_method == "Upload Video":
        # Step 3: Upload Video File
        uploaded_video = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi", "mkv"])

        if uploaded_video:
            with open("temp_video.mp4", "wb") as f:
                f.write(uploaded_video.read())

            # Process the video for pose detection
            st.write("### Processing the video...")
            process_video("temp_video.mp4", ideal_angles)
            st.write("Video processing complete.")
    else:
        # Step 3: Use Webcam
        st.write("### Using Webcam...")
        process_video(0, ideal_angles, is_webcam=True)
