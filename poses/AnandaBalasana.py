import numpy as np
import mediapipe as mp
import math

mp_pose = mp.solutions.pose

def calculate_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def detect_pose(landmarks):
    try:
        if landmarks:
            detected_pose = {
                "left_wrist": np.array([landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST.value].y]),
                "right_wrist": np.array([landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]),
                "left_shoulder": np.array([landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]),
                "right_shoulder": np.array([landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]),
                "left_hip": np.array([landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].y]),
                "right_hip": np.array([landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].y]),
                "left_knee": [landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].x, landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y],
                "right_knee": [landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].x, landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].y],
            }

            correct = [1] * len(landmarks.landmark)  # All landmarks marked correct initially
            feedback = ""

            left_wrist = detected_pose["left_wrist"]
            right_wrist = detected_pose["right_wrist"]
            left_knee = detected_pose["left_knee"]
            right_knee = detected_pose["right_knee"]

            # Hand and knee checks
            left_hand_knee_touch = np.linalg.norm(left_wrist - left_knee) < calculate_distance(
                detected_pose["left_hip"], detected_pose["right_hip"]) / 2
            right_hand_knee_touch = np.linalg.norm(right_wrist - right_knee) < calculate_distance(
                detected_pose["left_hip"], detected_pose["right_hip"]) / 2      

            # Hip and knee alignment checks
            if landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].y > landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE.value].y:
                correct[mp_pose.PoseLandmark.RIGHT_HIP.value] = 0
                feedback += "Lift your right hip higher. "

            if landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].y > landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE.value].y:
                correct[mp_pose.PoseLandmark.LEFT_HIP.value] = 0
                feedback += "Lift your left hip higher. "

            # Shoulder and wrist alignment checks
            if landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y > landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST.value].y:
                correct[mp_pose.PoseLandmark.LEFT_SHOULDER.value] = 0
                feedback += "Raise your left hand. "

            if landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y > landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST.value].y:
                correct[mp_pose.PoseLandmark.RIGHT_SHOULDER.value] = 0
                feedback += "Raise your right hand. "

            # Hand and knee touch validation
            if not (left_hand_knee_touch and right_hand_knee_touch):
                correct[mp_pose.PoseLandmark.LEFT_WRIST.value] = 0
                correct[mp_pose.PoseLandmark.RIGHT_WRIST.value] = 0
                feedback += "Make sure your hands are touching your knees. "

            accuracy = sum(correct) / len(correct) * 100
            pose_name = "Ardha Padmasana" if accuracy == 100 else "None"

            return accuracy, pose_name, correct, feedback

    except Exception as e:
        print(f"Error during pose detection: {e}")
        return 0.0, "Error", [0] * len(landmarks.landmark), ""
