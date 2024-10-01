import mediapipe as mp
import numpy as np
from .utils import calculate_angle, calculate_distance

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose


# Function to detect the desired pose
# Function to detect the desired pose
def detect_pose(landmarks):
    try:
        if landmarks:
            detected_pose = {
                "nose": np.array([landmarks.landmark[mp_pose.PoseLandmark.NOSE.value].x, landmarks.landmark[mp_pose.PoseLandmark.NOSE.value].y]),
                "left_eye_inner": np.array([landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE_INNER.value].x, landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE_INNER.value].y]),
                "left_eye": np.array([landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE.value].x, landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE.value].y]),
                "left_eye_outer": np.array([landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE_OUTER.value].x, landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE_OUTER.value].y]),
                "right_eye_inner": np.array([landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE_INNER.value].x, landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE_INNER.value].y]),
                "right_eye": np.array([landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE.value].x, landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE.value].y]),
                "right_eye_outer": np.array([landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE_OUTER.value].x, landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE_OUTER.value].y]),
                "left_ear": np.array([landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR.value].x, landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR.value].y]),
                "right_ear": np.array([landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EAR.value].x, landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EAR.value].y]),
                "mouth_left": np.array([landmarks.landmark[mp_pose.PoseLandmark.MOUTH_LEFT.value].x, landmarks.landmark[mp_pose.PoseLandmark.MOUTH_LEFT.value].y]),
                "mouth_right": np.array([landmarks.landmark[mp_pose.PoseLandmark.MOUTH_RIGHT.value].x, landmarks.landmark[mp_pose.PoseLandmark.MOUTH_RIGHT.value].y]),
                "left_shoulder": np.array([landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]),
                "right_shoulder": np.array([landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]),
                "left_elbow": np.array([landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]),
                "right_elbow": np.array([landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]),
                "left_wrist": np.array([landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST.value].y]),
                "right_wrist": np.array([landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]),
                "left_pinky": np.array([landmarks.landmark[mp_pose.PoseLandmark.LEFT_PINKY.value].x, landmarks.landmark[mp_pose.PoseLandmark.LEFT_PINKY.value].y]),
                "right_pinky": np.array([landmarks.landmark[mp_pose.PoseLandmark.RIGHT_PINKY.value].x, landmarks.landmark[mp_pose.PoseLandmark.RIGHT_PINKY.value].y]),
                "left_index": np.array([landmarks.landmark[mp_pose.PoseLandmark.LEFT_INDEX.value].x, landmarks.landmark[mp_pose.PoseLandmark.LEFT_INDEX.value].y]),
                "right_index": np.array([landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX.value].x, landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX.value].y]),
                "left_thumb": np.array([landmarks.landmark[mp_pose.PoseLandmark.LEFT_THUMB.value].x, landmarks.landmark[mp_pose.PoseLandmark.LEFT_THUMB.value].y]),
                "right_thumb": np.array([landmarks.landmark[mp_pose.PoseLandmark.RIGHT_THUMB.value].x, landmarks.landmark[mp_pose.PoseLandmark.RIGHT_THUMB.value].y]),
                "left_hip": np.array([landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].y]),
                "right_hip": np.array([landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].y]),
                "left_knee": np.array([landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE.value].y]),
                "right_knee": np.array([landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]),
                "left_ankle": np.array([landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]),
                "right_ankle": np.array([landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]),
                "left_heel": np.array([landmarks.landmark[mp_pose.PoseLandmark.LEFT_HEEL.value].x, landmarks.landmark[mp_pose.PoseLandmark.LEFT_HEEL.value].y]),
                "right_heel": np.array([landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HEEL.value].x, landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HEEL.value].y]),
                "left_foot_index": np.array([landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x, landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]),
                "right_foot_index": np.array([landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x, landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y])
            }

            # Set all landmarks as correct initially
            correct = [1] * len(landmarks.landmark)
            feedback = []

            # Check for specific pose conditions and mark incorrect landmarks
            left_shoulder_angle = calculate_angle(detected_pose["left_wrist"], detected_pose["left_shoulder"], detected_pose["left_hip"])
            if not (150 <= left_shoulder_angle <= 200):
                correct[mp_pose.PoseLandmark.LEFT_SHOULDER.value] = 0
                feedback.append("Adjust left shoulder angle.")

            right_shoulder_angle = calculate_angle(detected_pose["right_wrist"], detected_pose["right_shoulder"], detected_pose["right_hip"])
            if not (150 <= right_shoulder_angle <= 200):
                correct[mp_pose.PoseLandmark.RIGHT_SHOULDER.value] = 0
                feedback.append("Adjust right shoulder angle.")

            angle_left_hand = calculate_angle(detected_pose["left_shoulder"], detected_pose["left_elbow"], detected_pose['left_wrist'])
            if not (165 <= angle_left_hand <= 190):
                correct[mp_pose.PoseLandmark.RIGHT_WRIST.value] = 0
                feedback.append("Adjust left hand angle.")

            angle_right_hand = calculate_angle(detected_pose['right_shoulder'], detected_pose['right_elbow'], detected_pose['right_wrist'])
            if not (165 <= angle_right_hand <= 190):
                correct[mp_pose.PoseLandmark.LEFT_WRIST.value] = 0
                feedback.append("Adjust right hand angle.")

            # Legs
            if not ((landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].y < landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE.value].y) and 
                    (landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].y < landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y)):
                correct[mp_pose.PoseLandmark.LEFT_KNEE.value] = 0
                correct[mp_pose.PoseLandmark.RIGHT_KNEE.value] = 0
                feedback.append("Align hips and ankles properly.")

            left_leg_curve = calculate_angle(detected_pose["left_hip"], detected_pose["left_knee"], detected_pose["left_ankle"])
            if not (150 <= left_leg_curve <= 200):
                correct[mp_pose.PoseLandmark.LEFT_KNEE.value] = 0
                feedback.append("Adjust left leg curve.")

            right_leg_curve = calculate_angle(detected_pose["right_hip"], detected_pose["right_knee"], detected_pose["right_ankle"])
            if not (150 <= right_leg_curve <= 200):
                correct[mp_pose.PoseLandmark.RIGHT_KNEE.value] = 0
                feedback.append("Adjust right leg curve.")

            accuracy = sum(correct) / len(correct) * 100
            pose_name = "Utkatasana" if accuracy == 100 else "None"
            feedback_str = ' '.join(feedback) if feedback else "Pose is correct."

            return accuracy, pose_name, correct, feedback_str

    except Exception as e:
        print(f"Error during pose detection: {e}")
        return 0.0, "Error", [0] * len(landmarks.landmark), "Error in pose detection."
