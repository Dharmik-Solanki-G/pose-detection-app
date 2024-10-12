
from .utils import convert_landmarks_to_dict,calculate_joint_angles, classify_orientation, load_angles_from_csv

# Function to detect pose and validate angles.
def detect_pose(landmarks):
    feedback = []
    if not landmarks:
        print("No landmarks detected.")
        return 0.0, "No pose detected", [0] * 33, "No pose landmarks detected"

    ideal_angles = load_angles_from_csv(r"pose-detection-app\poses\CSV's\8 Uttitha ekapadasana (2).csv")
    correct = [1] * 33

    detected_pose = convert_landmarks_to_dict(landmarks)
    if detected_pose is None:
        return 0.0, "Error converting landmarks", [0] * 33, "Error accessing landmarks"

    print(f"Detected pose: {detected_pose}")

    if all(k in detected_pose for k in ['nose', 'left_shoulder', 'right_shoulder']):
        nose = detected_pose["nose"]
        left_shoulder = detected_pose["left_shoulder"]
        right_shoulder = detected_pose["right_shoulder"]

        direction = classify_orientation(nose, left_shoulder, right_shoulder)

        try:
            joint_angles = calculate_joint_angles(detected_pose)
        except Exception as e:
            print(f"Error calculating joint angles: {e}")
            return 0.0, "Error calculating joint angles", [0] * 33, str(e)

        print(f"Joint angles: {joint_angles}")

        angle_threshold = 20
        for key, angle in joint_angles.items():
            ideal_angle = ideal_angles.get(key, {}).get(direction, None)
            if ideal_angle is not None and not (ideal_angle - angle_threshold <= angle <= ideal_angle + angle_threshold):
                # Split key and adjust correctness tracking
                points = list(map(int, key.split('_')))
                if len(points) > 1:
                    correct[points[1]] = 0  # Mark this joint as incorrect
                feedback.append(
                    f"Angle at {key} should be between {ideal_angle - angle_threshold:.2f} "
                    f"and {ideal_angle + angle_threshold:.2f}, but is {angle:.2f}."
                )

        accuracy = sum(correct) / len(correct) * 100
        pose_name = "Correct" if accuracy == 100 else "Incorrect"
        feedback_str = ' '.join(feedback) if feedback else "Pose is correct"
        return accuracy, pose_name, correct, feedback_str

    return 0.0, "No pose detected", [0] * 33, "Missing critical landmarks"

