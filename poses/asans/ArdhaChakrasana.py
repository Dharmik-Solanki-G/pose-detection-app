import mediapipe as mp
import numpy as np
from .utils import calculate_angle, calculate_distance


# Initialize MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Define reference landmarks for the desired pose
reference_pose = {
    "left_wrist": np.array([0.5, 0.1]),
    "right_wrist": np.array([0.5, 0.1]),
    "left_shoulder": np.array([0.5, 0.3]),
    "right_shoulder": np.array([0.5, 0.3]),
    "left_hip": np.array([0.5, 0.7]),
    "right_hip": np.array([0.5, 0.7]),
}

# Function to detect the desired pose
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
                "eye": [landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE].x, landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE].y],
                "right_heel": [landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HEEL].x, landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HEEL].y],
            }

            # Set all landmarks as correct initially
            correct = [1] * len(landmarks.landmark)
            feedback = ""

            left_wrist = detected_pose["left_wrist"]
            right_wrist = detected_pose["right_wrist"]

            left_hand_hip_touch = np.linalg.norm(left_wrist - detected_pose["left_hip"]) < calculate_distance([landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].y], [landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].y]) * 7
            right_hand_hip_touch = np.linalg.norm(right_wrist - detected_pose["right_hip"]) < calculate_distance([landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].y], [landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].y]) * 7

            # Calculate similarity score and mark incorrect landmarks
            total_distance = 0.0
            for key in reference_pose.keys():
                detected_point = detected_pose.get(key)
                ref_point = reference_pose[key]
                distance = np.linalg.norm(detected_point - ref_point)
                total_distance += distance

            # Check for specific pose conditions and mark incorrect landmarks
            # Adjust these conditions based on the desired pose
            right_hip_y = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].y
            right_knee_y = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE.value].y
            if right_hip_y > right_knee_y:
                correct[mp_pose.PoseLandmark.RIGHT_HIP.value] = 0
                feedback += "Lower your right hip. "

            left_hip_y = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].y
            left_knee_y = landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE.value].y
            if left_hip_y > left_knee_y:
                correct[mp_pose.PoseLandmark.LEFT_HIP.value] = 0
                feedback += "Lower your left hip. "

            left_shoulder_y = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
            left_wrist_y = landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST.value].y
            if left_shoulder_y > left_wrist_y:
                correct[mp_pose.PoseLandmark.LEFT_SHOULDER.value] = 0
                feedback += "Raise your left hand upward. "

            right_shoulder_y = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
            right_wrist_y = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST.value].y
            if right_shoulder_y > right_wrist_y:
                correct[mp_pose.PoseLandmark.RIGHT_SHOULDER.value] = 0
                feedback += "Raise your right hand upward. "

            # If hands are not touching or hand angles are not correct, mark hand landmarks as incorrect
            if not (left_hand_hip_touch and right_hand_hip_touch):
                correct[mp_pose.PoseLandmark.LEFT_WRIST.value] = 0
                correct[mp_pose.PoseLandmark.RIGHT_WRIST.value] = 0
                feedback += "Make sure your hands are touching your hips. "

            # Calculate the angle between eye, right hip, and right heel
            curve = calculate_angle(detected_pose["eye"], detected_pose["right_hip"], detected_pose["right_heel"])
            if not curve < 175:
                correct[mp_pose.PoseLandmark.LEFT_HIP.value] = 0
                correct[mp_pose.PoseLandmark.RIGHT_HIP.value] = 0
                feedback += "Arch your back more to achieve the correct curve. "

            accuracy = sum(correct) / len(correct) * 100
            pose_name = "Ardha Chakrasana" if accuracy == 100 else "None"

            return accuracy, pose_name, correct, feedback.strip()

    except Exception as e:
        print(f"Error during pose detection: {e}")
        return 0.0, "Error", [0] * len(landmarks.landmark), str(e)
