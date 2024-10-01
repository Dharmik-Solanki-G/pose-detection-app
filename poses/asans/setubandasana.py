from .utils import calculate_angle, calculate_distance
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose.
mp_pose = mp.solutions.pose

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

            # Set all landmarks as correct initially
            correct = [1] * len(landmarks.landmark)
            feedback = []

            # Check if the hands are touching the ankles
            left_leg_hand_touch = calculate_distance(landmarks.landmark[16], landmarks.landmark[28]) < calculate_distance(landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER], landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]) / 2
            right_leg_hand_touch = calculate_distance(landmarks.landmark[15], landmarks.landmark[27]) < calculate_distance(landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER], landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]) / 2

            # If hips are not above the nose, mark as incorrect
            if landmarks.landmark[24].y > landmarks.landmark[0].y:
                correct[24] = 0
                feedback.append("Hips should be raised above the nose.")

            if landmarks.landmark[23].y > landmarks.landmark[0].y:
                correct[23] = 0
                feedback.append("Hips should be raised above the nose.")

            # If shoulders are below hips, mark as incorrect
            if landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y < landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].y:
                correct[mp_pose.PoseLandmark.LEFT_SHOULDER.value] = 0
                feedback.append("Shoulders should be above the hips.")

            if landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y < landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].y:
                correct[mp_pose.PoseLandmark.RIGHT_SHOULDER.value] = 0
                feedback.append("Shoulders should be above the hips.")

            # If hands are not touching ankles, mark hands and ankles as incorrect
            if not left_leg_hand_touch:
                correct[mp_pose.PoseLandmark.LEFT_WRIST.value] = 0
                correct[mp_pose.PoseLandmark.LEFT_ANKLE.value] = 0
                feedback.append("Left hand should be touching the left ankle.")

            if not right_leg_hand_touch:
                correct[mp_pose.PoseLandmark.RIGHT_WRIST.value] = 0
                correct[mp_pose.PoseLandmark.RIGHT_ANKLE.value] = 0
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


