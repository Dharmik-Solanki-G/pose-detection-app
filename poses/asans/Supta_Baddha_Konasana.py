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

            # Check if the feet are touching
            lags_touching = calculate_distance(detected_pose["right_ankle"], detected_pose["left_ankle"]) < calculate_distance(detected_pose["left_shoulder"], detected_pose["left_elbow"]) / 2
            if not lags_touching:
                correct[27] = 0  # LEFT_ANKLE
                correct[28] = 0  # RIGHT_ANKLE
                feedback.append("Feet not touching")

            # Check if the legs are properly aligned
            if detected_pose["mouth_left"][1] > detected_pose["left_ear"][1] or detected_pose["mouth_right"][1] > detected_pose["right_ear"][1]:
                correct[0] = 0  # NOSE
                correct[1] = 0  # LEFT_EYE_INNER
                correct[2] = 0  # LEFT_EYE
                correct[3] = 0  # LEFT_EYE_OUTER
                correct[4] = 0  # RIGHT_EYE_INNER
                correct[5] = 0  # RIGHT_EYE
                correct[6] = 0  # RIGHT_EYE_OUTER
                correct[7] = 0  # LEFT_EAR
                correct[8] = 0  # RIGHT_EAR
                correct[9] = 0  # MOUTH_LEFT
                correct[10] = 0  # MOUTH_RIGHT
                correct[11] = 0  # LEFT_SHOULDER
                correct[12] = 0  # RIGHT_SHOULDER
                feedback.append("Legs misaligned")

            if detected_pose["left_knee"][1] > detected_pose["left_ankle"][1]:
                correct[25] = 0  # LEFT_KNEE
                feedback.append("Left knee not in correct position")
            
            if detected_pose["right_knee"][1] > detected_pose["right_ankle"][1]:
                correct[26] = 0  # RIGHT_KNEE
                feedback.append("Right knee not in correct position")

            left_leg_curve = calculate_angle(detected_pose["left_hip"], detected_pose["left_knee"], detected_pose["left_ankle"])
            right_leg_curve = calculate_angle(detected_pose["right_hip"], detected_pose["right_knee"], detected_pose["right_ankle"])
            if (150 <= left_leg_curve <= 200) and (150 <= right_leg_curve <= 200):
                correct[25] = 0  # LEFT_KNEE
                correct[26] = 0  # RIGHT_KNEE
                feedback.append("Legs not bent correctly")

            accuracy = sum(correct) / len(correct) * 100
            pose_name = "Supta Baddha Konasana" if accuracy == 100 else "None"
            feedback_str = ', '.join(feedback) if feedback else "Correct Pose"

            return accuracy, pose_name, correct, feedback_str

    except Exception as e:
        print(f"Error during pose detection: {e}")
        return 0.0, "Error", [0] * len(landmarks), "Error"
