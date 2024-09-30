import cv2
import mediapipe as mp
import numpy as np
import math

# Initialize MediaPipe Pose.
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Define reference landmarks for Setubandasana
reference_pose = {
    "left_wrist": np.array([0.3, 0.85]),  # Wrists near the ankles, touching the legs
    "right_wrist": np.array([0.7, 0.85]),
    "left_shoulder": np.array([0.3, 0.6]),  # Shoulders lifted off the ground
    "right_shoulder": np.array([0.7, 0.6]),
    "left_hip": np.array([0.5, 0.65]),  # Hips raised
    "right_hip": np.array([0.5, 0.65]),
    "left_ankle": np.array([0.3, 1.0]),  # Feet on the ground
    "right_ankle": np.array([0.7, 1.0]),
    "head": np.array([0.5, 0.2])  # Head towards the ground
}

# Function to calculate the distance between two points
def calculate_distance(p1, p2):
    return np.linalg.norm(np.array([p1.x, p1.y]) - np.array([p2.x, p2.y]))

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

def detect_pose(landmarks):
    try:
        if landmarks:

            # Set all landmarks as correct initially
            correct = [1] * len(landmarks.landmark)
            feedback = []

            # Check if the hands are touching the ankles
            left_leg_hand_touch = calculate_distance(landmarks.landmark[16], landmarks.landmark[28]) < calculate_distance(landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER], landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]) / 2
            right_leg_hand_touch = calculate_distance(landmarks.landmark[15], landmarks.landmark[27]) < calculate_distance(landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER], landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]) / 2

            # If hips are not above the nose, mark as incorrect
            if landmarks.landmark[24].y > landmarks.landmark[0].y:
                correct[24] = 0
                feedback.append("Hips should be raised above the nose.")

            if landmarks.landmark[23].y > landmarks.landmark[0].y:
                correct[23] = 0
                feedback.append("Hips should be raised above the nose.")

            # If shoulders are below hips, mark as incorrect
            if landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y < landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].y:
                correct[mp_pose.PoseLandmark.LEFT_SHOULDER.value] = 0
                feedback.append("Shoulders should be above the hips.")

            if landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y < landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].y:
                correct[mp_pose.PoseLandmark.RIGHT_SHOULDER.value] = 0
                feedback.append("Shoulders should be above the hips.")

            # If hands are not touching ankles, mark hands and ankles as incorrect
            if not left_leg_hand_touch:
                correct[mp_pose.PoseLandmark.LEFT_WRIST.value] = 0
                correct[mp_pose.PoseLandmark.LEFT_ANKLE.value] = 0
                feedback.append("Left hand should be touching the left ankle.")

            if not right_leg_hand_touch:
                correct[mp_pose.PoseLandmark.RIGHT_WRIST.value] = 0
                correct[mp_pose.PoseLandmark.RIGHT_ANKLE.value] = 0
                feedback.append("Right hand should be touching the right ankle.")

            # Calculate final accuracy based on correct landmarks
            accuracy = sum(correct) / len(correct) * 100

            pose_name = "Setubandasana" if accuracy == 100 else "None"

            # Combine all feedback messages into a single string
            feedback_str = " | ".join(feedback) if feedback else "Pose is correct."

            return accuracy, pose_name, correct, feedback_str

    except Exception as e:
        print(f"Error during pose detection: {e}")
        return 0.0, "Error", "An error occurred during pose detection."


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
