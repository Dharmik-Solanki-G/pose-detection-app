import numpy as np
from .utils import calculate_angle, calculate_distance

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
                "left_pinky": np.array([landmarks[17]['x'], landmarks[17]['y']]),
                "right_pinky": np.array([landmarks[18]['x'], landmarks[18]['y']]),
                "left_index": np.array([landmarks[19]['x'], landmarks[19]['y']]),
                "right_index": np.array([landmarks[20]['x'], landmarks[20]['y']]),
                "left_thumb": np.array([landmarks[21]['x'], landmarks[21]['y']]),
                "right_thumb": np.array([landmarks[22]['x'], landmarks[22]['y']]),
                "left_hip": np.array([landmarks[23]['x'], landmarks[23]['y']]),
                "right_hip": np.array([landmarks[24]['x'], landmarks[24]['y']]),
                "left_knee": np.array([landmarks[25]['x'], landmarks[25]['y']]),
                "right_knee": np.array([landmarks[26]['x'], landmarks[26]['y']]),
                "left_ankle": np.array([landmarks[27]['x'], landmarks[27]['y']]),
                "right_ankle": np.array([landmarks[28]['x'], landmarks[28]['y']]),
                "left_heel": np.array([landmarks[29]['x'], landmarks[29]['y']]),
                "right_heel": np.array([landmarks[30]['x'], landmarks[30]['y']]),
                "left_foot_index": np.array([landmarks[31]['x'], landmarks[31]['y']]),
                "right_foot_index": np.array([landmarks[32]['x'], landmarks[32]['y']]),
            }

            # Initialize the correct array for 33 landmarks
            correct = [1] * len(detected_pose)  
            feedback = ""

            left_wrist = detected_pose["left_wrist"]
            right_wrist = detected_pose["right_wrist"]
            left_knee = detected_pose["left_knee"]
            right_knee = detected_pose["right_knee"]

            # Hand and knee checks
            left_hand_knee_touch = np.linalg.norm(left_wrist - left_knee) < calculate_distance(
                detected_pose["left_hip"], detected_pose["right_hip"]) / 2
            right_hand_knee_touch = np.linalg.norm(right_wrist - right_knee) < calculate_distance(
                detected_pose["left_hip"], detected_pose["right_hip"]) / 2      

            # Hip and knee alignment checks
            if detected_pose["right_hip"][1] > detected_pose["right_knee"][1]:
                correct[24] = 0
                feedback += "Lift your right hip higher. "

            if detected_pose["left_hip"][1] > detected_pose["left_knee"][1]:
                correct[23] = 0
                feedback += "Lift your left hip higher. "

            # Shoulder and wrist alignment checks
            if detected_pose["left_shoulder"][1] > detected_pose["left_wrist"][1]:
                correct[11] = 0
                feedback += "Raise your left hand. "

            if detected_pose["right_shoulder"][1] > detected_pose["right_wrist"][1]:
                correct[12] = 0
                feedback += "Raise your right hand. "

            # Hand and knee touch validation
            if not (left_hand_knee_touch and right_hand_knee_touch):
                correct[15] = 0
                correct[16] = 0
                feedback += "Make sure your hands are touching your knees. "

            accuracy = sum(correct) / len(correct) * 100
            pose_name = "Ardha Padmasana" if accuracy == 100 else "None"

            return accuracy, pose_name, correct, feedback

    except Exception as e:
        print(f"Error during pose detection: {e}")
        return 0.0, "Error", [0] * len(detected_pose), ""
