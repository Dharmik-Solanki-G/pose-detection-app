import numpy as np
from .utils import calculate_angle, calculate_distance

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
            feedback = []

            # Check for specific pose conditions and mark incorrect landmarks
            left_shoulder_y = landmarks[11]['y']
            left_elbow_y = landmarks[13]['y']
            if left_shoulder_y > left_elbow_y:
                correct[11] = 0
                feedback.append("Adjust left shoulder position")

            right_shoulder_y = landmarks[12]['y']
            right_elbow_y = landmarks[14]['y']
            if right_shoulder_y > right_elbow_y:
                correct[12] = 0
                feedback.append("Adjust right shoulder position")

            angle_left_hand = calculate_distance(detected_pose["right_index"], detected_pose["left_foot_index"]) < calculate_distance(detected_pose["left_shoulder"], detected_pose["left_elbow"]) / 1.2
            angle_right_hand = calculate_distance(detected_pose["left_index"], detected_pose["right_foot_index"]) < calculate_distance(detected_pose["left_shoulder"], detected_pose["left_elbow"]) / 1.2

            if not angle_left_hand:
                correct[15] = 0  # Left wrist
                correct[23] = 0  # Left hip
                feedback.append("Adjust left hand and hip position")

            if not angle_right_hand:
                correct[16] = 0  # Right wrist
                correct[24] = 0  # Right hip
                feedback.append("Adjust right hand and hip position")

            # Legs
            left_leg_curve = calculate_angle(detected_pose["left_hip"], detected_pose["left_knee"], detected_pose["left_ankle"])
            if not (150 <= left_leg_curve <= 200):
                correct[25] = 0  # Left knee
                feedback.append("Adjust left knee angle")

            right_leg_curve = calculate_angle(detected_pose["right_hip"], detected_pose["right_knee"], detected_pose["right_ankle"])
            if not (150 <= right_leg_curve <= 200):
                correct[26] = 0  # Right knee
                feedback.append("Adjust right knee angle")

            accuracy = sum(correct) / len(correct) * 100
            pose_name = "Paschimottanasana" if accuracy == 100 else "None"
            feedback_str = " | ".join(feedback) if feedback else "Pose is correct"

            return accuracy, pose_name, correct, feedback_str

    except Exception as e:
        print(f"Error during pose detection: {e}")
        return 0.0, "Error", [0] * len(landmarks), "Error in pose detection"
