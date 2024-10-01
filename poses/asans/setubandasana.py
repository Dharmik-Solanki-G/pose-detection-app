from .utils import calculate_angle, calculate_distance
import numpy as np

# Define reference landmarks for Setubandasana
reference_pose = {
    "left_wrist": np.array([0.3, 0.85]),  # Wrists near the ankles, touching the legs
    "right_wrist": np.array([0.7, 0.85]),
    "left_shoulder": np.array([0.3, 0.6]),  # Shoulders lifted off the ground
    "right_shoulder": np.array([0.7, 0.6]),
    "left_hip": np.array([0.5, 0.65]),  # Hips raised
    "right_hip": np.array([0.5, 0.65]),
    "left_ankle": np.array([0.3, 1.0]),  # Feet on the ground
    "right_ankle": np.array([0.7, 1.0]),
    "head": np.array([0.5, 0.2])  # Head towards the ground
}


def detect_pose(landmarks):
    try:
        if landmarks:

            # Define the detected landmarks for Setubandasana
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

            # Check if the hands are touching the ankles
            left_leg_hand_touch = calculate_distance(detected_pose["left_wrist"], detected_pose["left_ankle"]) < calculate_distance(detected_pose["left_shoulder"], detected_pose["left_hip"]) / 2
            right_leg_hand_touch = calculate_distance(detected_pose["right_wrist"], detected_pose["right_ankle"]) < calculate_distance(detected_pose["right_shoulder"], detected_pose["right_hip"]) / 2

            # If hips are not above the nose, mark as incorrect
            if detected_pose["left_hip"][1] <= detected_pose["nose"][1]:
                correct[23] = 0
                feedback.append("Left hip should be raised above the nose.")

            if detected_pose["right_hip"][1] <= detected_pose["nose"][1]:
                correct[24] = 0
                feedback.append("Right hip should be raised above the nose.")

            # If shoulders are below hips, mark as incorrect
            if detected_pose["left_shoulder"][1] < detected_pose["left_hip"][1]:
                correct[11] = 0
                feedback.append("Left shoulder should be above the left hip.")

            if detected_pose["right_shoulder"][1] < detected_pose["right_hip"][1]:
                correct[12] = 0
                feedback.append("Right shoulder should be above the right hip.")

            # If hands are not touching ankles, mark hands and ankles as incorrect
            if not left_leg_hand_touch:
                correct[15] = 0
                correct[27] = 0
                feedback.append("Left hand should be touching the left ankle.")

            if not right_leg_hand_touch:
                correct[16] = 0
                correct[28] = 0
                feedback.append("Right hand should be touching the right ankle.")

            # Calculate final accuracy based on correct landmarks
            accuracy = sum(correct) / len(correct) * 100

            pose_name = "Setubandasana" if accuracy == 100 else "None"

            # Combine all feedback messages into a single string
            feedback_str = " | ".join(feedback) if feedback else "Pose is correct."

            return accuracy, pose_name, correct, feedback_str

    except Exception as e:
        print(f"Error during pose detection: {e}")
        return 0.0, "Error", "An error occurred during pose detection."
