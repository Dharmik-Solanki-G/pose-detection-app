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
                "right_ankle": np.array([landmarks[28]['x'], landmarks[28]['y']])
            }

            # Initialize feedback list and correct array
            feedback = []
            correct = [1] * len(landmarks)

            # Check conditions and update feedback and correctness
            if landmarks[9]['y'] > landmarks[7]['y'] or landmarks[10]['y'] > landmarks[8]['y']:
                feedback.append("Head and shoulders should be aligned.")
                correct[0] = 0  # Nose
                correct[4] = 0  # Right Eye
                correct[5] = 0  # Right Eye Outer
                correct[6] = 0  # Right Eye Inner
                correct[2] = 0  # Left Eye
                correct[3] = 0  # Left Eye Outer
                correct[1] = 0  # Left Eye Inner
                correct[8] = 0  # Right Ear
                correct[7] = 0  # Left Ear
                correct[10] = 0  # Mouth Right
                correct[9] = 0  # Mouth Left
                correct[11] = 0  # Left Hip
                correct[12] = 0  # Right Hip
                correct[13] = 0  # Left Shoulder
                correct[14] = 0  # Right Shoulder

            right_heel_touching_left_knee = np.linalg.norm(detected_pose["right_heel"] - detected_pose["left_knee"]) < calculate_distance(detected_pose["left_shoulder"], detected_pose["right_shoulder"])
            left_heel_touching_right_knee = np.linalg.norm(detected_pose["left_heel"] - detected_pose["right_knee"]) < calculate_distance(detected_pose["left_shoulder"], detected_pose["right_shoulder"])
            if right_heel_touching_left_knee or left_heel_touching_right_knee:
                feedback.append("Right heel should not touch the left knee and vice versa.")
                correct[16] = 0  # Right Heel
                correct[25] = 0  # Left Knee
                correct[15] = 0  # Left Heel
                correct[26] = 0  # Right Knee

            left_hand_right_knee_touch = np.linalg.norm(detected_pose["left_index"] - detected_pose["right_knee"]) < calculate_distance(detected_pose["left_shoulder"], detected_pose["right_shoulder"])
            right_hand_left_knee_touch = np.linalg.norm(detected_pose["right_index"] - detected_pose["left_knee"]) < calculate_distance(detected_pose["left_shoulder"], detected_pose["right_shoulder"])
            if left_hand_right_knee_touch or right_hand_left_knee_touch:
                feedback.append("Left hand should not touch the right knee and vice versa.")
                correct[15] = 0  # Left Wrist
                correct[26] = 0  # Right Knee
                correct[25] = 0  # Left Knee
                correct[16] = 0  # Right Wrist

            angle_right_hand = calculate_angle(detected_pose['right_shoulder'], detected_pose['right_elbow'], detected_pose['right_wrist'])
            if not (165 <= angle_right_hand <= 190):
                feedback.append("Right hand should be in the correct angle.")
                correct[16] = 0  # Right Wrist

            accuracy = sum(correct) / len(correct) * 100
            pose_name = "Supta Matsyendrasana" if accuracy == 100 else "None"

            feedback_str = ' '.join(feedback) if feedback else 'Pose is correct'
            return accuracy, pose_name, correct, feedback_str

    except Exception as e:
        return 0.0, "Error", f"Error in pose detection: {str(e)}"
