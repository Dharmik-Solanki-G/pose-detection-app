import cv2
import mediapipe as mp
import numpy as np
import math

# Initialize MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Function to calculate the distance between two points
def calculate_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

# Function to calculate the angle between three points
def calculate_angle(point1, point2, point3):
    vec1 = [point1[0] - point2[0], point1[1] - point2[1]]
    vec2 = [point3[0] - point2[0], point3[1] - point2[1]]
    
    dot_product = vec1[0] * vec2[0] + vec1[1] * vec2[1]
    mag1 = math.sqrt(vec1[0]**2 + vec1[1]**2)
    mag2 = math.sqrt(vec2[0]**2 + vec2[1]**2)
    
    radians = math.acos(dot_product / (mag1 * mag2))
    angle = math.degrees(radians)
    
    return angle

# Function to draw landmarks and connections on the image
def draw_landmarks(image, landmarks, correct):
    h, w, _ = image.shape
    for idx, landmark in enumerate(landmarks.landmark):
        color = (0, 255, 0) if correct[idx] == 1 else (0, 0, 255)
        cx, cy = int(landmark.x * w), int(landmark.y * h)
        cv2.circle(image, (cx, cy), 5, color, -1)
    
    for connection in mp_pose.POSE_CONNECTIONS:
        start_idx, end_idx = connection
        if start_idx < len(landmarks.landmark) and end_idx < len(landmarks.landmark):
            start_point = landmarks.landmark[start_idx]
            end_point = landmarks.landmark[end_idx]
            color = (0, 255, 0) if correct[start_idx] == 1 and correct[end_idx] == 1 else (0, 0, 255)
            start = (int(start_point.x * w), int(start_point.y * h))
            end = (int(end_point.x * w), int(end_point.y * h))
            cv2.line(image, start, end, color, 2)



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
