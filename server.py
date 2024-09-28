import socketio
import json
from flask import Flask, render_template

# Create a Flask app
app = Flask(__name__)

# Create a Socket.IO server
sio = socketio.Server(cors_allowed_origins="*")
app.wsgi_app = socketio.WSGIApp(sio, app.wsgi_app)

from parvatasana import detect_parvatasana

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
    instruction = data.get('instructions', None)
    landmarks = data.get('landmarks', None)
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
