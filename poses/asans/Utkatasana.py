import numpy as np
from .utils import calculate_angle

# Function to detect the desired pose
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
            left_shoulder_angle = calculate_angle(detected_pose["left_wrist"], detected_pose["left_shoulder"], detected_pose["left_hip"])
            if not (150 <= left_shoulder_angle <= 200):
                correct[11] = 0  # LEFT_SHOULDER
                feedback.append("Adjust left shoulder angle.")

            right_shoulder_angle = calculate_angle(detected_pose["right_wrist"], detected_pose["right_shoulder"], detected_pose["right_hip"])
            if not (150 <= right_shoulder_angle <= 200):
                correct[12] = 0  # RIGHT_SHOULDER
                feedback.append("Adjust right shoulder angle.")

            angle_left_hand = calculate_angle(detected_pose["left_shoulder"], detected_pose["left_elbow"], detected_pose['left_wrist'])
            if not (165 <= angle_left_hand <= 190):
                correct[15] = 0  # LEFT_WRIST
                feedback.append("Adjust left hand angle.")

            angle_right_hand = calculate_angle(detected_pose['right_shoulder'], detected_pose['right_elbow'], detected_pose['right_wrist'])
            if not (165 <= angle_right_hand <= 190):
                correct[16] = 0  # RIGHT_WRIST
                feedback.append("Adjust right hand angle.")

            # Legs
            if not ((landmarks[23]['y'] < landmarks[27]['y']) and (landmarks[24]['y'] < landmarks[28]['y'])):
                correct[25] = 0  # LEFT_KNEE
                correct[26] = 0  # RIGHT_KNEE
                feedback.append("Align hips and ankles properly.")

            left_leg_curve = calculate_angle(detected_pose["left_hip"], detected_pose["left_knee"], detected_pose["left_ankle"])
            if not (150 <= left_leg_curve <= 200):
                correct[25] = 0  # LEFT_KNEE
                feedback.append("Adjust left leg curve.")

            right_leg_curve = calculate_angle(detected_pose["right_hip"], detected_pose["right_knee"], detected_pose["right_ankle"])
            if not (150 <= right_leg_curve <= 200):
                correct[26] = 0  # RIGHT_KNEE
                feedback.append("Adjust right leg curve.")

            accuracy = sum(correct) / len(correct) * 100
            pose_name = "Utkatasana" if accuracy == 100 else "None"
            feedback_str = ' '.join(feedback) if feedback else "Pose is correct."

            return accuracy, pose_name, correct, feedback_str

    except Exception as e:
        print(f"Error during pose detection: {e}")
        return 0.0, "Error", [0] * len(landmarks), "Error in pose detection."
