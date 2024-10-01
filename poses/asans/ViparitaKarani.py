from .utils import calculate_angle, calculate_distance
import numpy as np

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
            correct = [1] * len(detected_pose)
            feedback = []

            if detected_pose["right_elbow"][1] < detected_pose["right_wrist"][1]:
                correct[16] = 0  # right wrist
                correct[28] = 0  # right ankle
                feedback.append("Right wrist below elbow,")

            if detected_pose["left_elbow"][1] < detected_pose["left_wrist"][1]:
                correct[15] = 0  # left wrist
                correct[27] = 0  # left ankle
                feedback.append("Left wrist below elbow,")

            left_shoulder_y = detected_pose["left_shoulder"][1]
            left_hip_y = detected_pose["left_hip"][1]
            if left_shoulder_y < left_hip_y:
                correct[11] = 0  # left shoulder
                feedback.append("Left shoulder above hip,")

            right_shoulder_y = detected_pose["right_shoulder"][1]
            right_hip_y = detected_pose["right_hip"][1]
            if right_shoulder_y < right_hip_y:
                correct[12] = 0  # right shoulder
                feedback.append("Right shoulder above hip,")

            # Legs
            if detected_pose["left_hip"][1] < detected_pose["left_ankle"][1]:
                correct[25] = 0  # left knee
                correct[23] = 0  # left hip
                feedback.append("Left hip above ankle,")

            if detected_pose["right_hip"][1] < detected_pose["right_ankle"][1]:
                correct[26] = 0  # right knee
                correct[24] = 0  # right hip
                feedback.append("Right hip above ankle,")

            accuracy = sum(correct) / len(correct) * 100
            pose_name = "Viparita Karani" if accuracy == 100 else "None"
            feedback_str = " ".join(feedback) if feedback else "Good job!"

            return accuracy, pose_name, correct, feedback_str

    except Exception as e:
        print(f"Error during pose detection: {e}")
        return 0.0, "Error", [0] * len(landmarks), "Error"
