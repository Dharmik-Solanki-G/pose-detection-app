import socketio
from flask import Flask, render_template

# Create a Flask app
app = Flask(__name__)

# Create a Socket.IO server
sio = socketio.Server(cors_allowed_origins="*")
app.wsgi_app = socketio.WSGIApp(sio, app.wsgi_app)

# Function to detect pose and validate angles based on facing direction.
def detect_pose(results, ideal_angles, direction):
    feedback = []
    try:
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            correct = [1] * len(landmarks)

            joint_angles = calculate_joint_angles(landmarks)

            print("Calculated Angles from Video Frame:")
            for key, angle in joint_angles.items():
                print(f"{key}: {angle:.2f}")

            angle_threshold = 20

            # Check angles against thresholds based on direction.
            for key, angle in joint_angles.items():
                ideal_angle = ideal_angles[key][direction]
                if not (ideal_angle - angle_threshold <= angle <= ideal_angle + angle_threshold):
                    points = list(map(int, key.split('_')))
                    middle_point = points[1]  # The middle point is always the second in our angle keys
                    correct[middle_point] = 0
                    feedback.append(f"Angle at {key} should be between {ideal_angle - angle_threshold:.2f} and {ideal_angle + angle_threshold:.2f} degrees, but is {angle:.2f}.")

            accuracy = sum(correct) / len(correct) * 100
            pose_name = "Correct" if accuracy == 100 else "Incorrect"

            feedback_str = ' '.join(feedback) if feedback else "Pose is correct"
            return accuracy, pose_name, correct, feedback_str
        else:
            print("No pose landmarks detected in this frame.")
            return 0.0, "No pose detected", [], "No pose detected"

    except Exception as e:
        print(f"Error during pose detection: {e}")
        return 0.0, "Error", [], f"Error: {str(e)}"


# Flask route to serve the HTML (if needed)
@app.route('/')
def index():
    return render_template('index.html')

# Socket.IO event handler
@sio.event
def connect(sid, environ):
    print('Client connected:', sid)

@sio.event
def disconnect(sid):
    print('Client disconnected:', sid)

@sio.event
def poseData(sid, data):
    print(f'Received pose data from {sid}: {data}')
    # Handle or process the pose data here
    # For example, you can store it or perform further analysis

if __name__ == '__main__':
    # Run the app with eventlet, which is needed for Socket.IO
    import eventlet
    eventlet.wsgi.server(eventlet.listen(('', 5000)), app)
