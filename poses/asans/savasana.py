import numpy as np
from .utils import calculate_angle, calculate_distance

# Define reference landmarks for Savasana (Corpse Pose)
reference_pose = {
    "left_ankle": np.array([0.1, 0.9]),  # Left ankle position
    "right_ankle": np.array([0.9, 0.9]),  # Right ankle position
    "left_shoulder": np.array([0.3, 0.4]),  # Left shoulder position
    "right_shoulder": np.array([0.7, 0.4]),  # Right shoulder position
    "head": np.array([0.5, 0.2])  # Head position
}

def detect_pose(landmarks):
    if landmarks:
        # Define the detected landmarks for Savasana (Corpse Pose) using custom input
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

        # Check if the hands are touching the ankles
        left_leg_hand_touch = calculate_distance(detected_pose["left_wrist"], detected_pose["left_ankle"]) < calculate_distance(detected_pose["left_hip"], detected_pose["right_hip"]) * 4
        right_leg_hand_touch = calculate_distance(detected_pose["right_wrist"], detected_pose["right_ankle"]) < calculate_distance(detected_pose["left_hip"], detected_pose["right_hip"]) * 4

        # Calculate similarity score and mark incorrect landmarks
        total_distance = 0.0
        for key in reference_pose.keys():
            detected_point = detected_pose[key]
            ref_point = reference_pose[key]
            distance = np.linalg.norm(detected_point - ref_point)
            total_distance += distance
            if distance > 0.2:  # Mark as incorrect if distance is greater than 0.2
                index = list(reference_pose.keys()).index(key)  # Find the index of the key
                correct[index] = 0
                feedback.append(f"{key.replace('_', ' ').title()} position is incorrect")

        # If hands and legs are touching, mark as incorrect
        if left_leg_hand_touch:
            correct[15] = 0  # LEFT_WRIST
            correct[27] = 0  # LEFT_ANKLE
            feedback.append("Left hand should not touch left ankle")

        if right_leg_hand_touch:
            correct[16] = 0  # RIGHT_WRIST
            correct[28] = 0  # RIGHT_ANKLE
            feedback.append("Right hand should not touch right ankle")

        # Check body alignment (this logic can be adjusted as needed)
        if detected_pose["mouth_left"][1] > detected_pose["left_ear"][1] or detected_pose["mouth_right"][1] > detected_pose["right_ear"][1]:
            correct[0] = 0  # NOSE
            correct[4] = 0  # RIGHT_EYE
            correct[5] = 0  # RIGHT_EYE_OUTER
            correct[6] = 0  # RIGHT_EYE_INNER
            correct[2] = 0  # LEFT_EYE
            correct[3] = 0  # LEFT_EYE_OUTER
            correct[1] = 0  # LEFT_EYE_INNER
            correct[8] = 0  # RIGHT_EAR
            correct[7] = 0  # LEFT_EAR
            correct[9] = 0  # MOUTH_RIGHT
            correct[10] = 0  # MOUTH_LEFT
            correct[23] = 0  # LEFT_HIP
            correct[24] = 0  # RIGHT_HIP
            correct[11] = 0  # LEFT_SHOULDER
            correct[12] = 0  # RIGHT_SHOULDER
            feedback.append("Body alignment is incorrect")

        # Calculate accuracy based on correct landmarks
        accuracy = sum(correct) / len(correct) * 100
        pose_name = "Savasana (Corpse Pose)" if accuracy == 100 else "None"
        
        feedback_str = ", ".join(feedback) if feedback else "Pose is correct"

        return accuracy, pose_name, correct, feedback_str

    return 0.0, "None", [0] * len(landmarks), "No pose detected"
