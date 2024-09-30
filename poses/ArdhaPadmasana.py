import cv2
import mediapipe as mp
import numpy as np
import math

# Initialize MediaPipe Pose.
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Function to calculate the distance between two points
def calculate_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def calculate_angle(point1, point2, point3):
    # Calculate vectors
    vec1 = [point1[0] - point2[0], point1[1] - point2[1]]
    vec2 = [point3[0] - point2[0], point3[1] - point2[1]]
    
    # Calculate dot product
    dot_product = vec1[0] * vec2[0] + vec1[1] * vec2[1]
    
    # Calculate magnitudes
    mag1 = math.sqrt(vec1[0]**2 + vec1[1]**2)
    mag2 = math.sqrt(vec2[0]**2 + vec2[1]**2)
    
    # Calculate angle in radians
    radians = math.acos(dot_product / (mag1 * mag2))
    
    # Convert radians to degrees
    angle = math.degrees(radians)
    
    return angle

# Define reference landmarks for Parvatasana
reference_pose = {
    "left_wrist": np.array([0.5, 0.1]),
    "right_wrist": np.array([0.5, 0.1]),
    "left_shoulder": np.array([0.5, 0.3]),
    "right_shoulder": np.array([0.5, 0.3]),
    "left_hip": np.array([0.5, 0.7]),
    "right_hip": np.array([0.5, 0.7]),
}

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

# Function to detect pose and identify Parvatasana
def detect_pose(landmarks):
    feedback = []
    try:
        if landmarks:
            detected_pose = {
                "left_wrist": np.array([landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST.value].y]),
                "right_wrist": np.array([landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]),
                "left_shoulder": np.array([landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]),
                "right_shoulder": np.array([landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]),
                "left_hip": np.array([landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].y]),
                "right_hip": np.array([landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].y]),
                "left_knee": [landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].x, landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y],
                "right_knee": [landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].x, landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].y],
            }

            # Set all landmarks as correct initially
            correct = [1] * len(landmarks.landmark)

            left_wrist = detected_pose["left_wrist"]
            right_wrist = detected_pose["right_wrist"]
            left_knee = detected_pose["left_knee"]
            right_knee = detected_pose["right_knee"]
            left_hand_knee_touch = np.linalg.norm(left_wrist - left_knee) < calculate_distance([landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].y], [landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].y]) / 2
            right_hand_knee_touch = np.linalg.norm(right_wrist - right_knee) < calculate_distance([landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].y], [landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].y]) / 2

            # Calculate similarity score and mark incorrect landmarks
            total_distance = 0.0
            for key in reference_pose.keys():
                detected_point = detected_pose.get(key)
                ref_point = reference_pose[key]
                distance = np.linalg.norm(detected_point - ref_point)
                total_distance += distance

            right_hip_y = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].y
            right_knee_y = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE.value].y
            if right_hip_y > right_knee_y:
                correct[mp_pose.PoseLandmark.RIGHT_HIP.value] = 0
                feedback.append("Raise your right hip higher.")

            left_hip_y = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].y
            left_knee_y = landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE.value].y
            if left_hip_y > left_knee_y:
                correct[mp_pose.PoseLandmark.LEFT_HIP.value] = 0
                feedback.append("Raise your left hip higher.")

            left_shoulder_y = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
            left_wrist_y = landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST.value].y
            if left_shoulder_y > left_wrist_y:
                correct[mp_pose.PoseLandmark.LEFT_SHOULDER.value] = 0
                feedback.append("Lift your left hand higher.")

            right_shoulder_y = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
            right_wrist_y = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST.value].y
            if right_shoulder_y > right_wrist_y:
                correct[mp_pose.PoseLandmark.RIGHT_SHOULDER.value] = 0
                feedback.append("Lift your right hand higher.")

            # If hands are not touching or hand angles are not correct, mark hand landmarks as incorrect
            if not (left_hand_knee_touch and right_hand_knee_touch):
                correct[mp_pose.PoseLandmark.LEFT_WRIST.value] = 0
                correct[mp_pose.PoseLandmark.RIGHT_WRIST.value] = 0
                feedback.append("Ensure both hands are touching the respective knees.")

            accuracy = sum(correct) / len(correct) * 100
            pose_name = "Ardha Padmasana" if accuracy == 100 else "None"

            feedback_str = " ".join(feedback) if feedback else "Good job!"
            return accuracy, pose_name, correct, feedback_str

    except Exception as e:
        print(f"Error during pose detection: {e}")
        return 0.0, "Error", [0] * len(landmarks.landmark), str(e)

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
