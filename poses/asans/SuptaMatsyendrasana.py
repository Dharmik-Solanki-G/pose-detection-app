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

            # Initialize feedback list and correct array
            feedback = []
            correct = [1] * len(landmarks.landmark)

            # Check conditions and update feedback and correctness
            if landmarks.landmark[mp_pose.PoseLandmark.MOUTH_LEFT.value].y > landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR.value].y or landmarks.landmark[mp_pose.PoseLandmark.MOUTH_RIGHT.value].y > landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EAR.value].y:
                feedback.append("Head and shoulders should be aligned.")
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

            right_heel_touching_left_knee = np.linalg.norm(detected_pose["right_heel"] - detected_pose["left_knee"]) < calculate_distance(detected_pose["left_shoulder"], detected_pose["right_shoulder"])
            left_heel_touching_right_knee = np.linalg.norm(detected_pose["left_heel"] - detected_pose["right_knee"]) < calculate_distance(detected_pose["left_shoulder"], detected_pose["right_shoulder"])
            if right_heel_touching_left_knee or left_heel_touching_right_knee:
                feedback.append("Right heel should not touch the left knee and vice versa.")
                correct[mp_pose.PoseLandmark.RIGHT_HEEL.value] = 0
                correct[mp_pose.PoseLandmark.LEFT_KNEE.value] = 0
                correct[mp_pose.PoseLandmark.LEFT_HEEL.value] = 0
                correct[mp_pose.PoseLandmark.RIGHT_KNEE.value] = 0

            left_hand_right_knee_touch = np.linalg.norm(detected_pose["left_index"] - detected_pose["right_knee"]) < calculate_distance(detected_pose["left_shoulder"], detected_pose["right_shoulder"])
            right_hand_left_knee_touch = np.linalg.norm(detected_pose["right_index"] - detected_pose["left_knee"]) < calculate_distance(detected_pose["left_shoulder"], detected_pose["right_shoulder"])
            if left_hand_right_knee_touch or right_hand_left_knee_touch:
                feedback.append("Left hand should not touch the right knee and vice versa.")
                correct[mp_pose.PoseLandmark.LEFT_WRIST.value] = 0
                correct[mp_pose.PoseLandmark.RIGHT_KNEE.value] = 0
                correct[mp_pose.PoseLandmark.LEFT_KNEE.value] = 0
                correct[mp_pose.PoseLandmark.RIGHT_WRIST.value] = 0

            angle_right_hand = calculate_angle(detected_pose['right_shoulder'], detected_pose['right_elbow'], detected_pose['right_wrist'])
            if not (165 <= angle_right_hand <= 190):
                feedback.append("Right hand should be in the correct angle.")
                correct[mp_pose.PoseLandmark.RIGHT_WRIST.value] = 0

            accuracy = sum(correct) / len(correct) * 100
            pose_name = "Supta Matsyendrasana" if accuracy == 100 else "None"

            feedback_str = ' '.join(feedback) if feedback else 'Pose is correct'
            return accuracy, pose_name, correct, feedback_str

    except Exception as e:
        return 0.0, "Error", f"Error in pose detection: {str(e)}"

