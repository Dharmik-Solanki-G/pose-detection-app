from .utils import calculate_angle, calculate_distance
import numpy as np

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

            correct = [1] * len(landmarks)
            feedback_str = []

            left_knee_y = landmarks[25]['y']
            left_wrist_y = landmarks[15]['y']
            if left_knee_y < left_wrist_y:
                correct[11] = 0  # left_shoulder
                feedback_str.append("Left shoulder is incorrect")

            right_knee_y = landmarks[26]['y']
            right_wrist_y = landmarks[16]['y']
            if right_knee_y < right_wrist_y:
                correct[12] = 0  # right_shoulder
                feedback_str.append("Right shoulder is incorrect")

            left_wrist = [landmarks[15]['x'], landmarks[15]['y']]
            angle_left_hand = calculate_distance(left_wrist, detected_pose["left_knee"]) < calculate_distance([landmarks[23]['x'], landmarks[23]['y']], [landmarks[24]['x'], landmarks[24]['y']]) * 3

            right_wrist = [landmarks[16]['x'], landmarks[16]['y']]
            angle_right_hand = calculate_distance(right_wrist, detected_pose["right_knee"]) < calculate_distance([landmarks[23]['x'], landmarks[23]['y']], [landmarks[24]['x'], landmarks[24]['y']]) * 3

            if not (angle_left_hand and angle_right_hand):
                correct[15] = 0  # left_wrist
                correct[16] = 0  # right_wrist
                feedback_str.append("Hands are incorrectly positioned")

            right_leg_fold = calculate_distance(detected_pose["right_ankle"], detected_pose["right_hip"]) < calculate_distance([landmarks[23]['x'], landmarks[23]['y']], [landmarks[24]['x'], landmarks[24]['y']]) * 3
            left_leg_fold = calculate_distance(detected_pose["left_ankle"], detected_pose["left_hip"]) < calculate_distance([landmarks[23]['x'], landmarks[23]['y']], [landmarks[24]['x'], landmarks[24]['y']]) * 3

            if not (right_leg_fold and left_leg_fold):
                correct[23] = 0  # left_hip
                correct[24] = 0  # right_hip
                feedback_str.append("Legs are incorrectly positioned")

            accuracy = sum(correct) / len(correct) * 100
            pose_name = "Vajrasana" if accuracy == 100 else "None"

            return accuracy, pose_name, correct, " | ".join(feedback_str)

    except Exception as e:
        print(f"Error during pose detection: {e}")
        return 0.0, "Error", [0] * len(landmarks), "Error"
