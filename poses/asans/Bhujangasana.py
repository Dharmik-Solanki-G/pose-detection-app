import numpy as np
from .utils import calculate_angle

def detect_pose(landmarks):
    try:
        if landmarks:
            detected_pose = {
                "nose": np.array([landmarks[0]['x'], landmarks[0]['y']]),
                "left_eye_inner": np.array([landmarks[1]['x'], landmarks[1]['y']]),
                "left_eye": np.array([landmarks[2]['x'], landmarks[2]['y']]),
                "left_eye_outer": np.array([landmarks[3]['x'], landmarks[3]['y']]),
                "right_eye_inner": np.array([landmarks[4]['x'], landmarks[4]['y']]),
                "right_eye": np.array([landmarks[5]['x'], landmarks[5]['y']]),
                "right_eye_outer": np.array([landmarks[6]['x'], landmarks[6]['y']]),
                "left_ear": np.array([landmarks[7]['x'], landmarks[7]['y']]),
                "right_ear": np.array([landmarks[8]['x'], landmarks[8]['y']]),
                "mouth_left": np.array([landmarks[9]['x'], landmarks[9]['y']]),
                "mouth_right": np.array([landmarks[10]['x'], landmarks[10]['y']]),
                "left_shoulder": np.array([landmarks[11]['x'], landmarks[11]['y']]),
                "right_shoulder": np.array([landmarks[12]['x'], landmarks[12]['y']]),
                "left_elbow": np.array([landmarks[13]['x'], landmarks[13]['y']]),
                "right_elbow": np.array([landmarks[14]['x'], landmarks[14]['y']]),
                "left_wrist": np.array([landmarks[15]['x'], landmarks[15]['y']]),
                "right_wrist": np.array([landmarks[16]['x'], landmarks[16]['y']]),
                "left_hip": np.array([landmarks[23]['x'], landmarks[23]['y']]),
                "right_hip": np.array([landmarks[24]['x'], landmarks[24]['y']]),
                "left_knee": np.array([landmarks[25]['x'], landmarks[25]['y']]),
                "right_knee": np.array([landmarks[26]['x'], landmarks[26]['y']]),
                "left_ankle": np.array([landmarks[27]['x'], landmarks[27]['y']]),
                "right_ankle": np.array([landmarks[28]['x'], landmarks[28]['y']]),
            }

            # Set all landmarks as correct initially
            correct = [1] * len(landmarks)
            error_messages = []

            # Check for specific pose conditions and mark incorrect landmarks
            left_shoulder_y = landmarks[11]['y']
            left_elbow_y = landmarks[13]['y']
            if left_shoulder_y > left_elbow_y:
                correct[11] = 0
                error_messages.append("Left shoulder is below the elbow.")

            right_shoulder_y = landmarks[12]['y']
            right_elbow_y = landmarks[14]['y']
            if right_shoulder_y > right_elbow_y:
                correct[12] = 0
                error_messages.append("Right shoulder is below the elbow.")

            right_mouth_y = landmarks[10]['y']
            if right_shoulder_y < right_mouth_y:
                correct[8] = 0  # Right ear
                correct[7] = 0  # Left ear
                correct[10] = 0  # Right mouth
                correct[9] = 0  # Left mouth
                correct[0] = 0  # Nose
                error_messages.append("Head is not aligned with shoulders.")

            # Legs
            left_leg_curve = calculate_angle(detected_pose["left_hip"], detected_pose["left_knee"], detected_pose["left_ankle"])
            if not (150 <= left_leg_curve <= 200):
                correct[25] = 0  # Left knee
                error_messages.append("Left knee angle is not correct.")

            right_leg_curve = calculate_angle(detected_pose["right_hip"], detected_pose["right_knee"], detected_pose["right_ankle"])
            if not (150 <= right_leg_curve <= 200):
                correct[26] = 0  # Right knee
                error_messages.append("Right knee angle is not correct.")

            accuracy = sum(correct) / len(correct) * 100
            pose_name = "Bhujangasana" if accuracy == 100 else "None"
            error_message = " ".join(error_messages) if error_messages else "Pose is correct."

            return accuracy, pose_name, correct, error_message

    except Exception as e:
        print(f"Error during pose detection: {e}")
        return 0.0, "Error", [0] * len(landmarks), "Error"
