import cv2
import mediapipe as mp
import numpy as np
import math

# Initialize MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Define reference landmarks for Vrksasana (Tree Pose)
reference_pose = {
    "left_wrist": np.array([0.5, 0.1]),
    "right_wrist": np.array([0.5, 0.1]),
    "left_shoulder": np.array([0.5, 0.3]),
    "right_shoulder": np.array([0.5, 0.3]),
    "left_hip": np.array([0.5, 0.7]),
    "right_hip": np.array([0.5, 0.7]),
}

def calculate_angle(point1, point2, point3):
    # Calculate vectors
    vec1 = [point1[0] - point2[0], point1[1] - point2[1]]
    vec2 = [point3[0] - point2[0], point3[1] - point2[1]]
    
    # Calculate dot product
    dot_product = vec1[0] * vec2[0] + vec1[1] * vec2[1]
    
    # Calculate magnitudes
    mag1 = math.sqrt(vec1[0]**2 + vec1[1]**2)
    mag2 = math.sqrt(vec2[0]**2 + vec2[1]**2)
    
    # Handle division by zero
    if mag1 == 0 or mag2 == 0:
        return 0.0
    
    # Calculate angle in radians
    radians = math.acos(dot_product / (mag1 * mag2))
    
    # Convert radians to degrees
    angle = math.degrees(radians)
    
    return angle

# Function to calculate the distance between two points
def calculate_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

# Function to draw landmarks and connections on the image
def draw_landmarks(image, landmarks, correct):
    h, w, _ = image.shape
    for idx, landmark in enumerate(landmarks.landmark):
        color = (0, 255, 0) if correct[idx] == 1 else (0, 0, 255)
        cx, cy = int(landmark.x * w), int(landmark.y * h)
        cv2.circle(image, (cx, cy), 5, color, -1)
    
    for connection in mp_pose.POSE_CONNECTIONS:
        start_idx, end_idx = connection
        start_point = landmarks.landmark[start_idx]
        end_point = landmarks.landmark[end_idx]
        color = (0, 255, 0) if correct[start_idx] == 1 and correct[end_idx] == 1 else (0, 0, 255)
        start = (int(start_point.x * w), int(start_point.y * h))
        end = (int(end_point.x * w), int(end_point.y * h))
        cv2.line(image, start, end, color, 2)

def calculate_arm_angle(landmarks, side):
    if side == "left":
        shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y
        elbow = landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].x, landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y
        wrist = landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x, landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y
    else:
        shoulder = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y
        elbow = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].x, landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y
        wrist = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x, landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y
    
    return calculate_angle(shoulder, elbow, wrist)
def detect_pose(landmarks):
    try:
        if landmarks:
            detected_pose = {
                "left_wrist": [landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x, landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y],
                "right_wrist": [landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x, landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y],
                "left_shoulder": [landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y],
                "right_shoulder": [landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y],
                "left_hip": [landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x, landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y],
                "right_hip": [landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].x, landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y],
                "left_foot": [landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x, landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y],
                "right_foot": [landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].x, landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y],
                "left_knee": [landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].x, landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y],
                "right_knee": [landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].x, landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].y],
            }

            # Set all landmarks as correct initially
            correct = [1] * len(landmarks.landmark)
            feedback = []

            left_wrist = detected_pose["left_wrist"]
            right_wrist = detected_pose["right_wrist"]
            hand_touch = np.linalg.norm(np.array(left_wrist) - np.array(right_wrist)) < calculate_distance(detected_pose["left_hip"], detected_pose["right_hip"]) / 1.2

            left_knee_index = detected_pose["left_knee"]
            right_foot_index = detected_pose["right_foot"]
            right_foot_knee_not_touch = calculate_distance(left_knee_index, right_foot_index) > calculate_distance(detected_pose["left_hip"], detected_pose["right_hip"]) * 2.5

            right_knee_index = detected_pose["right_knee"]
            left_foot_index = detected_pose["left_foot"]
            left_foot_knee_not_touch = calculate_distance(right_knee_index, left_foot_index) > calculate_distance(detected_pose["left_hip"], detected_pose["right_hip"]) * 2.5

            # Calculate similarity score and mark incorrect landmarks
            total_distance = 0.0
            for key in reference_pose.keys():
                detected_point = detected_pose.get(key)
                ref_point = reference_pose[key]
                distance = np.linalg.norm(np.array(detected_point) - ref_point)
                total_distance += distance

                if landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y < landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y:
                    correct[mp_pose.PoseLandmark.LEFT_SHOULDER.value] = 0
                    feedback.append("Left shoulder too low")

                if landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y < landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y:
                    correct[mp_pose.PoseLandmark.RIGHT_SHOULDER.value] = 0
                    feedback.append("Right shoulder too low")

            angle_left_hand = calculate_arm_angle(landmarks, "left")
            angle_right_hand = calculate_arm_angle(landmarks, "right")

            # If hands are not touching or angles are not correct, mark hand landmarks as incorrect
            if not hand_touch or angle_left_hand < 160 or angle_right_hand < 160:
                correct[mp_pose.PoseLandmark.LEFT_WRIST.value] = 0
                correct[mp_pose.PoseLandmark.RIGHT_WRIST.value] = 0
                feedback.append("Hands not touching or angles incorrect")

            # If feet or knees are not correctly positioned, update the feedback
            if not right_foot_knee_not_touch or not left_foot_knee_not_touch:
                correct[mp_pose.PoseLandmark.RIGHT_HIP.value] = 1
                correct[mp_pose.PoseLandmark.LEFT_HIP.value] = 1

            accuracy = (sum(correct) / len(correct)) * 100
            pose_name = "Vrksasana" if accuracy > 80 else "None"

            # Combine feedback list into a single string
            feedback_str = " | ".join(feedback) if feedback else "Pose looks good"

            return accuracy, pose_name, correct, feedback_str

    except Exception as e:
        print(f"Error during pose detection: {e}")
        return None, 0.0, [0] * len(landmarks.landmark), "Error"

# Main function to capture video and detect pose
def main():
    cap = cv2.VideoCapture(0)  # Use 0 for the default webcam
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Ignoring empty camera frame.")
                continue

            # Convert the BGR image to RGB and process it with MediaPipe Pose
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if results.pose_landmarks:
                accuracy, pose_name, correct, feedback_str = detect_pose(results.pose_landmarks)
                draw_landmarks(frame, results.pose_landmarks, correct)
                cv2.putText(frame, f'Pose: {pose_name}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, f'Accuracy: {accuracy:.2f}%', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, f'Feedback: {feedback_str}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow('Pose Detection', frame)

            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
