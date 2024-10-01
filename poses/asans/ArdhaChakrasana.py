import numpy as np
from .utils import calculate_angle, calculate_distance

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
            feedback = ""

            # Check for hand-to-hip touch
            left_hand_hip_touch = np.linalg.norm(detected_pose["left_wrist"] - detected_pose["left_hip"]) < calculate_distance(detected_pose["left_hip"], detected_pose["right_hip"]) * 0.7
            right_hand_hip_touch = np.linalg.norm(detected_pose["right_wrist"] - detected_pose["right_hip"]) < calculate_distance(detected_pose["left_hip"], detected_pose["right_hip"]) * 0.7

            # Calculate similarity score and mark incorrect landmarks
            total_distance = 0.0
            for key in reference_pose.keys():
                detected_point = detected_pose.get(key)
                ref_point = reference_pose[key]
                if detected_point is not None:
                    distance = np.linalg.norm(detected_point - ref_point)
                    total_distance += distance

            # Specific pose checks and feedback
            right_hip_y = landmarks[24]['y']
            right_knee_y = landmarks[26]['y']
            if right_hip_y > right_knee_y:
                correct[24] = 0
                feedback += "Lower your right hip. "

            left_hip_y = landmarks[23]['y']
            left_knee_y = landmarks[25]['y']
            if left_hip_y > left_knee_y:
                correct[23] = 0
                feedback += "Lower your left hip. "

            left_shoulder_y = landmarks[11]['y']
            left_wrist_y = landmarks[15]['y']
            if left_shoulder_y > left_wrist_y:
                correct[11] = 0
                feedback += "Raise your left hand upward. "

            right_shoulder_y = landmarks[12]['y']
            right_wrist_y = landmarks[16]['y']
            if right_shoulder_y > right_wrist_y:
                correct[12] = 0
                feedback += "Raise your right hand upward. "

            # Hand touch conditions
            if not (left_hand_hip_touch and right_hand_hip_touch):
                correct[15] = 0  # Left wrist
                correct[16] = 0  # Right wrist
                feedback += "Make sure your hands are touching your hips. "

            # Calculate the curve angle
            curve = calculate_angle(detected_pose["nose"], detected_pose["right_hip"], detected_pose["right_knee"])
            if not curve < 175:
                correct[23] = 0  # Left hip
                correct[24] = 0  # Right hip
                feedback += "Arch your back more to achieve the correct curve. "

            accuracy = sum(correct) / len(correct) * 100
            pose_name = "Ardha Chakrasana" if accuracy == 100 else "None"

            return accuracy, pose_name, correct, feedback.strip()

    except Exception as e:
        print(f"Error during pose detection: {e}")
        return 0.0, "Error", [0] * len(landmarks), str(e)
