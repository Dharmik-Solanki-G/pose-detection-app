from .utils import calculate_angle, calculate_distance
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose


def detect_pose(landmarks):
    try:
        if landmarks:
            detected_pose = {
                "nose": np.array([landmarks.landmark[mp_pose.PoseLandmark.NOSE.value].x, landmarks.landmark[mp_pose.PoseLandmark.NOSE.value].y]),
                "left_shoulder": np.array([landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]),
                "right_shoulder": np.array([landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]),
                "left_elbow": np.array([landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]),
                "right_elbow": np.array([landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]),
                "left_wrist": np.array([landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST.value].y]),
                "right_wrist": np.array([landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]),
                "left_hip": np.array([landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].y]),
                "right_hip": np.array([landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].y]),
                "left_knee": np.array([landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE.value].y]),
                "right_knee": np.array([landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]),
                "left_ankle": np.array([landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]),
                "right_ankle": np.array([landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y])
            }

            # Set all landmarks as correct initially
            correct = [1] * len(landmarks.landmark)
            feedback = []

            if landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y < landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST.value].y:
                correct[mp_pose.PoseLandmark.RIGHT_WRIST.value] = 0
                correct[mp_pose.PoseLandmark.RIGHT_ANKLE.value] = 0
                feedback.append("Right wrist below elbow,")

            if landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW.value].y < landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST.value].y:
                correct[mp_pose.PoseLandmark.LEFT_WRIST.value] = 0
                correct[mp_pose.PoseLandmark.LEFT_ANKLE.value] = 0
                feedback.append("Left wrist below elbow,")

            left_shoulder_y = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
            left_hip_y = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].y
            if left_shoulder_y < left_hip_y:
                correct[mp_pose.PoseLandmark.LEFT_SHOULDER.value] = 0
                feedback.append("Left shoulder above hip,")

            right_shoulder_y = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
            right_hip_y = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].y
            if right_shoulder_y < right_hip_y:
                correct[mp_pose.PoseLandmark.RIGHT_SHOULDER.value] = 0
                feedback.append("Right shoulder above hip,")

            # Legs
            if landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].y < landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE.value].y:
                correct[mp_pose.PoseLandmark.LEFT_KNEE.value] = 0
                correct[mp_pose.PoseLandmark.LEFT_HIP.value] = 0
                feedback.append("Left hip above ankle,")

            if landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].y < landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y:
                correct[mp_pose.PoseLandmark.RIGHT_KNEE.value] = 0
                correct[mp_pose.PoseLandmark.RIGHT_HIP.value] = 0
                feedback.append("Right hip above ankle,")

            accuracy = sum(correct) / len(correct) * 100
            pose_name = "Viparita Karani" if accuracy == 100 else "None"
            feedback_str = " ".join(feedback) if feedback else "Good job!"

            return accuracy, pose_name, correct, feedback_str

    except Exception as e:
        print(f"Error during pose detection: {e}")
        return 0.0, "Error", [0] * len(landmarks.landmark), "Error"

