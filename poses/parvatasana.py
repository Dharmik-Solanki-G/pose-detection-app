import numpy as np
import math
from .utils import calculate_angle, calculate_distance

reference_pose = {
    "left_wrist": np.array([0.5, 0.1]),
    "right_wrist": np.array([0.5, 0.1]),
    "left_shoulder": np.array([0.5, 0.3]),
    "right_shoulder": np.array([0.5, 0.3]),
    "left_hip": np.array([0.5, 0.7]),
    "right_hip": np.array([0.5, 0.7]),
}

def detect_parvatasana(landmarks):
    try:
        if landmarks:
            detected_pose = {
                "left_wrist": np.array([landmarks[15]['x'],
                                        landmarks[15]['y']]),
                "right_wrist": np.array([landmarks[16]['x'],
                                         landmarks[16]['y']]),
                "left_shoulder": np.array([landmarks[11]['x'],
                                           landmarks[11]['y']]),
                "right_shoulder": np.array([landmarks[12]['x'],
                                            landmarks[12]['y']]),
                "left_hip": np.array([landmarks[23]['x'],
                                      landmarks[23]['y']]),
                "right_hip": np.array([landmarks[24]['x'],
                                       landmarks[24]['y']]),
            }
            feedback = []

            # Set all landmarks as correct initially
            correct = [1] * len(landmarks)

            left_wrist = detected_pose["left_wrist"]
            right_wrist = detected_pose["right_wrist"]
            hand_touch = np.linalg.norm(left_wrist - right_wrist) < calculate_distance([landmarks[23]['x'], landmarks[23]['y']], [landmarks[24]['x'], landmarks[24]['y']]) / 2

            # Calculate similarity score and mark incorrect landmarks
            total_distance = 0.0
            for key in reference_pose.keys():
                detected_point = detected_pose.get(key)
                ref_point = reference_pose[key]
                distance = np.linalg.norm(detected_point - ref_point)
                total_distance += distance

            right_hip_y = landmarks[24]['y']
            right_knee_y = landmarks[26]['y']
            if right_hip_y > right_knee_y:
                correct[24] = 0
                feedback.append("Right hip should be above the knee.")

            left_hip_y = landmarks[23]['y']
            left_knee_y = landmarks[25]['y']
            if left_hip_y > left_knee_y:
                correct[23] = 0
                feedback.append("Left hip should be above the knee.")

            left_shoulder = [landmarks[11]['x'], landmarks[11]['y']]
            left_elbow = [landmarks[13]['x'], landmarks[13]['y']]
            left_wrist = [landmarks[15]['x'], landmarks[15]['y']]
            angle_left_hand = calculate_angle(left_shoulder, left_elbow, left_wrist)

            left_shoulder_y = landmarks[11]['y']
            left_wrist_y = landmarks[15]['y']

            if left_shoulder_y < left_wrist_y:
                correct[11] = 0
                feedback.append("Left shoulder should be higher than the wrist.")

            right_shoulder = [landmarks[12]['x'], landmarks[12]['y']]
            right_elbow = [landmarks[14]['x'], landmarks[14]['y']]
            right_wrist = [landmarks[16]['x'], landmarks[16]['y']]
            angle_right_hand = calculate_angle(right_shoulder, right_elbow, right_wrist)
            right_shoulder_y = landmarks[12]['y']
            right_wrist_y = landmarks[16]['y']

            if right_shoulder_y < right_wrist_y:
                correct[12] = 0
                feedback.append("Right shoulder should be higher than the wrist.")

            # If hands are not touching or hand angles are not correct, mark hand landmarks as incorrect
            if not hand_touch or angle_left_hand < 160 or angle_right_hand < 160:
                correct[15] = 0
                correct[16] = 0
                feedback.append("Hands should be touching and the angle should be correct.")

            accuracy = sum(correct) / len(correct) * 100
            pose_name = "Parvatasana" if accuracy == 100 else "None"
            feedback_str = ' '.join(feedback) if feedback else "Pose is correct"

            return accuracy, pose_name, correct , feedback_str

    except Exception as e:
        print(f"Error during pose detection: {e}")
        return 0.0, "Error", [0] * len(landmarks) , ""