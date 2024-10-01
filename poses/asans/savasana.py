import mediapipe as mp
import numpy as np
from .utils import calculate_angle, calculate_distance

# Initialize MediaPipe Pose.
mp_pose = mp.solutions.pose

# Define reference landmarks for Savasana (Corpse Pose)
reference_pose = {
    "left_ankle": np.array([0.1, 0.9]),  # Left ankle position
    "right_ankle": np.array([0.9, 0.9]),  # Right ankle position
    "left_shoulder": np.array([0.3, 0.4]),  # Left shoulder position
    "right_shoulder": np.array([0.7, 0.4]),  # Right shoulder position
    "head": np.array([0.5, 0.2])  # Head position
}
def detect_pose(pose_landmarks):
    if pose_landmarks:
        # Define the detected landmarks for Savasana (Corpse Pose)
        detected_pose = {
            "left_ankle": np.array([pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                                    pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]),
            "right_ankle": np.array([pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                                    pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]),
            "left_shoulder": np.array([pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                    pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]),
            "right_shoulder": np.array([pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                        pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]),
            "head": np.array([pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE.value].x,
                            pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE.value].y])
        }

        # Set all landmarks as correct initially
        correct = [1] * len(pose_landmarks.landmark)
        feedback = []

        # Check if the hands are touching the ankles
        left_leg_hand_touch = calculate_distance(pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST],
                                                 pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]) < calculate_distance(pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP], pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]) * 4
        right_leg_hand_touch = calculate_distance(pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST],
                                                  pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]) < calculate_distance(pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP], pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]) * 4

        # Calculate similarity score and mark incorrect landmarks
        total_distance = 0.0
        for key, idx in zip(reference_pose.keys(), range(len(pose_landmarks.landmark))):
            detected_point = detected_pose[key]
            ref_point = reference_pose[key]
            distance = np.linalg.norm(detected_point - ref_point)
            total_distance += distance
            if distance > 0.2:  # Mark as incorrect if distance is greater than 0.2
                correct[idx] = 0
                feedback.append(f"{key.replace('_', ' ').title()} position is incorrect")

        # If hands and legs are touching, mark as incorrect
        if left_leg_hand_touch:
            correct[mp_pose.PoseLandmark.LEFT_WRIST.value] = 0
            correct[mp_pose.PoseLandmark.LEFT_ANKLE.value] = 0
            feedback.append("Left hand should not touch left ankle")

        if right_leg_hand_touch:
            correct[mp_pose.PoseLandmark.RIGHT_WRIST.value] = 0
            correct[mp_pose.PoseLandmark.RIGHT_ANKLE.value] = 0
            feedback.append("Right hand should not touch right ankle")

        if pose_landmarks.landmark[mp_pose.PoseLandmark.MOUTH_LEFT.value].y > pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR.value].y or pose_landmarks.landmark[mp_pose.PoseLandmark.MOUTH_RIGHT.value].y > pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EAR.value].y:
            correct[mp_pose.PoseLandmark.NOSE.value] = 0
            correct[mp_pose.PoseLandmark.RIGHT_EYE.value] = 0
            correct[mp_pose.PoseLandmark.RIGHT_EYE_OUTER.value] = 0
            correct[mp_pose.PoseLandmark.RIGHT_EYE_INNER.value] = 0
            correct[mp_pose.PoseLandmark.LEFT_EYE.value] = 0
            correct[mp_pose.PoseLandmark.LEFT_EYE_OUTER.value] = 0
            correct[mp_pose.PoseLandmark.LEFT_EYE_INNER.value] = 0
            correct[mp_pose.PoseLandmark.RIGHT_EAR.value] = 0
            correct[mp_pose.PoseLandmark.LEFT_EAR.value] = 0
            correct[mp_pose.PoseLandmark.MOUTH_RIGHT.value] = 0
            correct[mp_pose.PoseLandmark.MOUTH_LEFT.value] = 0
            correct[mp_pose.PoseLandmark.LEFT_HIP.value] = 0
            correct[mp_pose.PoseLandmark.RIGHT_HIP.value] = 0
            correct[mp_pose.PoseLandmark.LEFT_SHOULDER.value] = 0
            correct[mp_pose.PoseLandmark.RIGHT_SHOULDER.value] = 0
            feedback.append("Body alignment is incorrect")

        # Calculate accuracy based on correct landmarks
        accuracy = sum(correct) / len(correct) * 100
        pose_name = "Savasana (Corpse Pose)" if accuracy == 100 else "None"
        
        feedback_str = ", ".join(feedback) if feedback else "Pose is correct"

        return accuracy, pose_name, correct, feedback_str

    return 0.0, "None", [0] * len(pose_landmarks.landmark), "No pose detected"

