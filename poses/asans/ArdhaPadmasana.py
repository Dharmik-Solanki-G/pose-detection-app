import mediapipe as mp
import numpy as np
from .utils import calculate_angle, calculate_distance

# Initialize MediaPipe Pose.
mp_pose = mp.solutions.pose


# Define reference landmarks for Parvatasana
reference_pose = {
    "left_wrist": np.array([0.5, 0.1]),
    "right_wrist": np.array([0.5, 0.1]),
    "left_shoulder": np.array([0.5, 0.3]),
    "right_shoulder": np.array([0.5, 0.3]),
    "left_hip": np.array([0.5, 0.7]),
    "right_hip": np.array([0.5, 0.7]),
}


# Function to detect pose and identify Parvatasana
def detect_pose(landmarks):
    feedback = []
    try:
        if landmarks:
            detected_pose = {
                "left_wrist": np.array([landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST.value].y]),
                "right_wrist": np.array([landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]),
                "left_shoulder": np.array([landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]),
                "right_shoulder": np.array([landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]),
                "left_hip": np.array([landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].y]),
                "right_hip": np.array([landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].y]),
                "left_knee": [landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].x, landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y],
                "right_knee": [landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].x, landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].y],
            }

            # Set all landmarks as correct initially
            correct = [1] * len(landmarks.landmark)

            left_wrist = detected_pose["left_wrist"]
            right_wrist = detected_pose["right_wrist"]
            left_knee = detected_pose["left_knee"]
            right_knee = detected_pose["right_knee"]
            left_hand_knee_touch = np.linalg.norm(left_wrist - left_knee) < calculate_distance([landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].y], [landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].y]) / 2
            right_hand_knee_touch = np.linalg.norm(right_wrist - right_knee) < calculate_distance([landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].y], [landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].y]) / 2

            # Calculate similarity score and mark incorrect landmarks
            total_distance = 0.0
            for key in reference_pose.keys():
                detected_point = detected_pose.get(key)
                ref_point = reference_pose[key]
                distance = np.linalg.norm(detected_point - ref_point)
                total_distance += distance

            right_hip_y = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].y
            right_knee_y = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE.value].y
            if right_hip_y > right_knee_y:
                correct[mp_pose.PoseLandmark.RIGHT_HIP.value] = 0
                feedback.append("Raise your right hip higher.")

            left_hip_y = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].y
            left_knee_y = landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE.value].y
            if left_hip_y > left_knee_y:
                correct[mp_pose.PoseLandmark.LEFT_HIP.value] = 0
                feedback.append("Raise your left hip higher.")

            left_shoulder_y = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
            left_wrist_y = landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST.value].y
            if left_shoulder_y > left_wrist_y:
                correct[mp_pose.PoseLandmark.LEFT_SHOULDER.value] = 0
                feedback.append("Lift your left hand higher.")

            right_shoulder_y = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
            right_wrist_y = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST.value].y
            if right_shoulder_y > right_wrist_y:
                correct[mp_pose.PoseLandmark.RIGHT_SHOULDER.value] = 0
                feedback.append("Lift your right hand higher.")

            # If hands are not touching or hand angles are not correct, mark hand landmarks as incorrect
            if not (left_hand_knee_touch and right_hand_knee_touch):
                correct[mp_pose.PoseLandmark.LEFT_WRIST.value] = 0
                correct[mp_pose.PoseLandmark.RIGHT_WRIST.value] = 0
                feedback.append("Ensure both hands are touching the respective knees.")

            accuracy = sum(correct) / len(correct) * 100
            pose_name = "Ardha Padmasana" if accuracy == 100 else "None"

            feedback_str = " ".join(feedback) if feedback else "Good job!"
            return accuracy, pose_name, correct, feedback_str

    except Exception as e:
        print(f"Error during pose detection: {e}")
        return 0.0, "Error", [0] * len(landmarks.landmark), str(e)
