import mediapipe as mp
import numpy as np
from .utils import calculate_angle, calculate_distance

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose

# Define reference landmarks for Vrksasana (Tree Pose)
reference_pose = {
    "left_wrist": np.array([0.5, 0.1]),
    "right_wrist": np.array([0.5, 0.1]),
    "left_shoulder": np.array([0.5, 0.3]),
    "right_shoulder": np.array([0.5, 0.3]),
    "left_hip": np.array([0.5, 0.7]),
    "right_hip": np.array([0.5, 0.7]),
}


def calculate_arm_angle(landmarks, side):
    if side == "left":
        shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y
        elbow = landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].x, landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y
        wrist = landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x, landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y
    else:
        shoulder = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y
        elbow = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].x, landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y
        wrist = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x, landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y
    
    return calculate_angle(shoulder, elbow, wrist)
def detect_pose(landmarks):
    try:
        if landmarks:
            detected_pose = {
                "left_wrist": [landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x, landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y],
                "right_wrist": [landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x, landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y],
                "left_shoulder": [landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y],
                "right_shoulder": [landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y],
                "left_hip": [landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x, landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y],
                "right_hip": [landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].x, landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y],
                "left_foot": [landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x, landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y],
                "right_foot": [landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].x, landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y],
                "left_knee": [landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].x, landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y],
                "right_knee": [landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].x, landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].y],
            }

            # Set all landmarks as correct initially
            correct = [1] * len(landmarks.landmark)
            feedback = []

            left_wrist = detected_pose["left_wrist"]
            right_wrist = detected_pose["right_wrist"]
            hand_touch = np.linalg.norm(np.array(left_wrist) - np.array(right_wrist)) < calculate_distance(detected_pose["left_hip"], detected_pose["right_hip"]) / 1.2

            left_knee_index = detected_pose["left_knee"]
            right_foot_index = detected_pose["right_foot"]
            right_foot_knee_not_touch = calculate_distance(left_knee_index, right_foot_index) > calculate_distance(detected_pose["left_hip"], detected_pose["right_hip"]) * 2.5

            right_knee_index = detected_pose["right_knee"]
            left_foot_index = detected_pose["left_foot"]
            left_foot_knee_not_touch = calculate_distance(right_knee_index, left_foot_index) > calculate_distance(detected_pose["left_hip"], detected_pose["right_hip"]) * 2.5

            # Calculate similarity score and mark incorrect landmarks
            total_distance = 0.0
            for key in reference_pose.keys():
                detected_point = detected_pose.get(key)
                ref_point = reference_pose[key]
                distance = np.linalg.norm(np.array(detected_point) - ref_point)
                total_distance += distance

                if landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y < landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y:
                    correct[mp_pose.PoseLandmark.LEFT_SHOULDER.value] = 0
                    feedback.append("Left shoulder too low")

                if landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y < landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y:
                    correct[mp_pose.PoseLandmark.RIGHT_SHOULDER.value] = 0
                    feedback.append("Right shoulder too low")

            angle_left_hand = calculate_arm_angle(landmarks, "left")
            angle_right_hand = calculate_arm_angle(landmarks, "right")

            # If hands are not touching or angles are not correct, mark hand landmarks as incorrect
            if not hand_touch or angle_left_hand < 160 or angle_right_hand < 160:
                correct[mp_pose.PoseLandmark.LEFT_WRIST.value] = 0
                correct[mp_pose.PoseLandmark.RIGHT_WRIST.value] = 0
                feedback.append("Hands not touching or angles incorrect")

            # If feet or knees are not correctly positioned, update the feedback
            if not right_foot_knee_not_touch or not left_foot_knee_not_touch:
                correct[mp_pose.PoseLandmark.RIGHT_HIP.value] = 1
                correct[mp_pose.PoseLandmark.LEFT_HIP.value] = 1

            accuracy = (sum(correct) / len(correct)) * 100
            pose_name = "Vrksasana" if accuracy > 80 else "None"

            # Combine feedback list into a single string
            feedback_str = " | ".join(feedback) if feedback else "Pose looks good"

            return accuracy, pose_name, correct, feedback_str

    except Exception as e:
        print(f"Error during pose detection: {e}")
        return None, 0.0, [0] * len(landmarks.landmark), "Error"
