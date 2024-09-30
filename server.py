import socketio
from flask import Flask, send_from_directory, render_template

# Create a Flask app
app = Flask(__name__, static_folder='public', static_url_path='')

# Create a Socket.IO server
sio = socketio.Server(cors_allowed_origins="*")
app.wsgi_app = socketio.WSGIApp(sio, app.wsgi_app)

from poses.parvatasana import detect_parvatasana

# Flask route to serve the HTML file
@app.route('/')
def index():
    return app.send_static_file('index.html')

# Route to serve static files from the public directory
@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('public', filename)

# Socket.IO event handler
@sio.event
def connect(sid, environ):
    print('Client connected:', sid)

@sio.event
def disconnect(sid):
    print('Client disconnected:', sid)

@sio.event
def poseData(sid, data):
    instruction = data.get('instructions', None)
    landmarks = data.get('pose_landmarks', None)
    height = data.get('height', None)
    width = data.get('width', None)

    # Process the pose data
    accuracy, text, correct = detect_parvatasana(landmarks[0])
    
    # Emit feedback back to the client
    sio.emit('poseFeedback', {'accuracy': accuracy, 'text': text, 'correct': correct}, room=sid)

if __name__ == '__main__':
    # Run the app with eventlet, which is needed for Socket.IO
    import eventlet
    eventlet.wsgi.server(eventlet.listen(('', 5000)), app)
