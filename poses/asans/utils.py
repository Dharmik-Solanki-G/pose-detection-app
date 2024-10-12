import numpy as np
import math
import pandas as pd


def calculate_angle(point1, point2, point3):
    """Calculate the angle between three points."""
    try:
        # Calculate vectors
        vec1 = [point1[0] - point2[0], point1[1] - point2[1]]
        vec2 = [point3[0] - point2[0], point3[1] - point2[1]]

        # Calculate dot product and magnitudes
        dot_product = vec1[0] * vec2[0] + vec1[1] * vec2[1]
        mag1 = math.sqrt(vec1[0]**2 + vec1[1]**2)
        mag2 = math.sqrt(vec2[0]**2 + vec2[1]**2)

        # Prevent division-by-zero errors
        if mag1 == 0 or mag2 == 0:
            print("Degenerate vector found, returning 0 degrees")
            return 0.0

        # Calculate the angle in radians
        radians = math.acos(dot_product / (mag1 * mag2))

        # Convert radians to degrees
        return math.degrees(radians)

    except Exception as e:
        print(f"Error calculating angle: {e}")
        return 0.0  # Return 0 degrees on error


def calculate_distance(p1, p2):
    """Calculate the Euclidean distance between two points."""
    return np.linalg.norm(np.array(p1) - np.array(p2))

def classify_orientation(nose, left_shoulder, right_shoulder):
    """Classify the orientation based on nose and shoulders."""
    # Midpoint between the shoulders
    shoulder_midpoint = [
        (left_shoulder[0] + right_shoulder[0]) / 2,
        (left_shoulder[1] + right_shoulder[1]) / 2,
    ]

    # Calculate angle between nose and shoulder midpoint
    dx = nose[0] - shoulder_midpoint[0]
    dy = nose[1] - shoulder_midpoint[1]
    angle = np.arctan2(dy, dx) * 180.0 / np.pi

    # Determine orientation based on the angle
    if 70 <= abs(angle) <= 110:
        return "Front"
    elif nose[0] < left_shoulder[0]:
        return "Left"
    elif nose[0] > right_shoulder[0]:
        return "Right"
    else:
        return "Uncertain"

def calculate_joint_angles(landmarks_dict):
    """Calculate specific joint angles based on given landmarks."""
    angles = {}

    def get_angle(point1, point2, point3):
        """Helper function to calculate angle between three points."""
        vec1 = point1 - point2  # Vector from point2 to point1
        vec2 = point3 - point2  # Vector from point2 to point3

        mag1 = np.linalg.norm(vec1)
        mag2 = np.linalg.norm(vec2)

        if mag1 == 0 or mag2 == 0:
            print("Degenerate vector found, setting angle to 0 degrees.")
            return 0.0

        cos_angle = np.dot(vec1, vec2) / (mag1 * mag2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Avoid numerical errors
        return np.degrees(np.arccos(cos_angle))

    try:
        # Calculate angles for left and right elbows
        angles['15_13_11'] = get_angle(
            landmarks_dict['left_wrist'], 
            landmarks_dict['left_elbow'], 
            landmarks_dict['left_shoulder']
        )
        angles['12_14_16'] = get_angle(
            landmarks_dict['right_shoulder'], 
            landmarks_dict['right_elbow'], 
            landmarks_dict['right_wrist']
        )
    except KeyError as e:
        print(f"Missing landmark during angle calculation: {e}")
    
    print(f"Calculated angles: {angles}")
    return angles

def load_angles_from_csv(csv_path):
    """Load ideal angles from a CSV file."""
    df = pd.read_csv(csv_path)
    angles_dict = {}
    for _, row in df.iterrows():
        angles_dict[row['Landmark Pair']] = {
            'Front': row['Front (degrees)'],
            'Right': row['Right (degrees)'],
            'Left': row['Left (degrees)']
        }
    return angles_dict

# Function to convert Mediapipe landmarks into a dictionary of NumPy arrays.
def convert_landmarks_to_dict(landmarks):
    try:
        return {
            "nose": np.array([landmarks[0].x, landmarks[0].y]),
            "left_eye_inner": np.array([landmarks[1].x, landmarks[1].y]),
            "left_eye": np.array([landmarks[2].x, landmarks[2].y]),
            "left_eye_outer": np.array([landmarks[3].x, landmarks[3].y]),
            "right_eye_inner": np.array([landmarks[4].x, landmarks[4].y]),
            "right_eye": np.array([landmarks[5].x, landmarks[5].y]),
            "right_eye_outer": np.array([landmarks[6].x, landmarks[6].y]),
            "left_ear": np.array([landmarks[7].x, landmarks[7].y]),
            "right_ear": np.array([landmarks[8].x, landmarks[8].y]),
            "mouth_left": np.array([landmarks[9].x, landmarks[9].y]),
            "mouth_right": np.array([landmarks[10].x, landmarks[10].y]),
            "left_shoulder": np.array([landmarks[11].x, landmarks[11].y]),
            "right_shoulder": np.array([landmarks[12].x, landmarks[12].y]),
            "left_elbow": np.array([landmarks[13].x, landmarks[13].y]),
            "right_elbow": np.array([landmarks[14].x, landmarks[14].y]),
            "left_wrist": np.array([landmarks[15].x, landmarks[15].y]),
            "right_wrist": np.array([landmarks[16].x, landmarks[16].y]),
            "left_pinky": np.array([landmarks[17].x, landmarks[17].y]),
            "right_pinky": np.array([landmarks[18].x, landmarks[18].y]),
            "left_index": np.array([landmarks[19].x, landmarks[19].y]),
            "right_index": np.array([landmarks[20].x, landmarks[20].y]),
            "left_thumb": np.array([landmarks[21].x, landmarks[21].y]),
            "right_thumb": np.array([landmarks[22].x, landmarks[22].y]),
            "left_hip": np.array([landmarks[23].x, landmarks[23].y]),
            "right_hip": np.array([landmarks[24].x, landmarks[24].y]),
            "left_knee": np.array([landmarks[25].x, landmarks[25].y]),
            "right_knee": np.array([landmarks[26].x, landmarks[26].y]),
            "left_ankle": np.array([landmarks[27].x, landmarks[27].y]),
            "right_ankle": np.array([landmarks[28].x, landmarks[28].y]),
            "left_heel": np.array([landmarks[29].x, landmarks[29].y]),
            "right_heel": np.array([landmarks[30].x, landmarks[30].y]),
            "left_foot_index": np.array([landmarks[31].x, landmarks[31].y]),
            "right_foot_index": np.array([landmarks[32].x, landmarks[32].y]),
        }
    except Exception as e:
        print(f"Error converting landmarks: {e}")
        return None