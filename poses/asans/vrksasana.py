import numpy as np
from .utils import calculate_angle, calculate_distance

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
        shoulder = landmarks["left_shoulder"]
        elbow = landmarks["left_elbow"]
        wrist = landmarks["left_wrist"]
    else:
        shoulder = landmarks["right_shoulder"]
        elbow = landmarks["right_elbow"]
        wrist = landmarks["right_wrist"]
    
    return calculate_angle(shoulder, elbow, wrist)

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
            correct = [1] * len(detected_pose)
            feedback = []

            left_wrist = detected_pose["left_wrist"]
            right_wrist = detected_pose["right_wrist"]
            hand_touch = np.linalg.norm(left_wrist - right_wrist) < calculate_distance(detected_pose["left_hip"], detected_pose["right_hip"]) / 1.2

            left_knee_index = detected_pose["left_knee"]
            right_foot_index = detected_pose["right_ankle"]
            right_foot_knee_not_touch = calculate_distance(left_knee_index, right_foot_index) > calculate_distance(detected_pose["left_hip"], detected_pose["right_hip"]) * 2.5

            right_knee_index = detected_pose["right_knee"]
            left_foot_index = detected_pose["left_ankle"]
            left_foot_knee_not_touch = calculate_distance(right_knee_index, left_foot_index) > calculate_distance(detected_pose["left_hip"], detected_pose["right_hip"]) * 2.5

            # Calculate similarity score and mark incorrect landmarks
            total_distance = 0.0
            for key in reference_pose.keys():
                detected_point = detected_pose.get(key)
                ref_point = reference_pose[key]
                distance = np.linalg.norm(detected_point - ref_point)
                total_distance += distance

                if detected_pose["left_shoulder"][1] < detected_pose["left_wrist"][1]:
                    correct[11] = 0  # Index for LEFT_SHOULDER
                    feedback.append("Left shoulder too low")

                if detected_pose["right_shoulder"][1] < detected_pose["right_wrist"][1]:
                    correct[12] = 0  # Index for RIGHT_SHOULDER
                    feedback.append("Right shoulder too low")

            angle_left_hand = calculate_arm_angle(detected_pose, "left")
            angle_right_hand = calculate_arm_angle(detected_pose, "right")

            # If hands are not touching or angles are not correct, mark hand landmarks as incorrect
            if not hand_touch or angle_left_hand < 160 or angle_right_hand < 160:
                correct[15] = 0  # Index for LEFT_WRIST
                correct[16] = 0  # Index for RIGHT_WRIST
                feedback.append("Hands not touching or angles incorrect")

            # If feet or knees are not correctly positioned, update the feedback
            if not right_foot_knee_not_touch or not left_foot_knee_not_touch:
                correct[23] = 1  # Index for RIGHT_HIP
                correct[24] = 1  # Index for LEFT_HIP

            accuracy = (sum(correct) / len(correct)) * 100
            pose_name = "Vrksasana" if accuracy > 80 else "None"

            # Combine feedback list into a single string
            feedback_str = " | ".join(feedback) if feedback else "Pose looks good"

            return accuracy, pose_name, correct, feedback_str

    except Exception as e:
        print(f"Error during pose detection: {e}")
        return None, 0.0, [0] * len(detected_pose), "Error"
