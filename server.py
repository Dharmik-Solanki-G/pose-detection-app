import socketio
from flask import Flask, send_from_directory

# Create a Flask app
app = Flask(__name__, static_folder='public', static_url_path='')

# Create a Socket.IO server
sio = socketio.Server(cors_allowed_origins="*")
app.wsgi_app = socketio.WSGIApp(sio, app.wsgi_app)

# Import the PoseDetection class
from poses.pose_detection import PoseDetection

# Create an instance of the PoseDetection class
pose_detector = PoseDetection()

# Flask route to serve the HTML file
@app.route('/')
def index():
    return app.send_static_file('index.html')

# Route to serve static files from the public directory
@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('public', filename)

# Socket.IO event handler for connecting clients
@sio.event
def connect(sid, environ):
    print('Client connected:', sid)

# Socket.IO event handler for disconnecting clients
@sio.event
def disconnect(sid):
    print('Client disconnected:', sid)

# Socket.IO event handler for processing pose data
@sio.event
def poseData(sid, data):
    instructions = data.get('instructions', None)  # Extract instructions
    landmarks = data.get('pose_landmarks', None)
    print(instructions)

    # Check if landmarks are provided
    if landmarks:
        # Process the pose data using the PoseDetection wrapper
        accuracy, pose_name, correct , feedback_str = pose_detector.analyze_pose(instructions, landmarks[0])  # Pass instructions along

        # Emit feedback back to the client
        sio.emit('poseFeedback', {'accuracy': accuracy, 'text': pose_name, 'correct': correct , "feedback" : feedback_str}, room=sid)
    else:
        print("No landmarks provided.")

if __name__ == '__main__':
    # Run the app with eventlet, which is needed for Socket.IO
    import eventlet
    eventlet.wsgi.server(eventlet.listen(('', 5000)), app)
