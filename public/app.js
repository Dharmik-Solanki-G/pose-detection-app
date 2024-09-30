import PoseDetection from './poseDetection.js';

// Connect to the WebSocket server
const socket = io('http://localhost:5000'); // Make sure to match your server address

// Function to start the webcam
async function setupWebcam() {
    const video = document.querySelector('.local-video');
    const stream = await navigator.mediaDevices.getUserMedia({
        video: true
    });
    video.srcObject = stream;
    return stream;
}

async function startPoseDetection() {
    // Get webcam stream
    const stream = await setupWebcam();

    // Create PoseDetection instance and start detection
    const poseDetection = new PoseDetection(stream);
    let poseResult;  // Variable to hold the latest pose result

    // Set a callback function to process the results
    poseDetection.setResultCallback((result, canvasWidth, canvasHeight) => {
        poseResult = result; // Store the latest pose result
    });

    // Send pose data every 1 second
    setInterval(() => {
        if (poseResult) {
            // Emit the latest pose data to the WebSocket server
            socket.emit('poseData', { pose_landmarks: poseResult.landmarks });
        }
    }, 1000); // 1000 milliseconds = 1 second

    socket.on('poseFeedback', (data) => { handleDetection(data); });

    async function handleDetection(data) {
        const { accuracy, correct, correct2, feedback } = data;

        // Set correct points in PoseDetection
        poseDetection.setCorrectPoints(correct, correct2);

        // Display accuracy and feedback on feedback canvas
        displayFeedback(accuracy, feedback);
    }

    // Function to display accuracy and feedback
    function displayFeedback(accuracy, feedback) {
        const canvas = document.getElementById('feedback-canva');
        const ctx = canvas.getContext('2d');

        // Clear the canvas before drawing
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Set a semi-transparent background for the canvas
        ctx.fillStyle = 'rgba(255, 255, 255, 0.8)'; // White with slight transparency
        ctx.fillRect(0, 0, canvas.width, canvas.height); // Rectangle background

        // Set styles for text
        ctx.fillStyle = '#333'; // Darker text color for contrast
        ctx.font = 'bold 20px Arial'; // Bold font for emphasis

        // Draw the accuracy
        ctx.fillText(`Accuracy: ${accuracy.toFixed(2)}%`, 10, 30); // Adjusted padding

        // Set a different style for the feedback text
        ctx.font = '16px Arial'; // Regular font for feedback
        ctx.fillStyle = '#6200ea'; // Purple color for feedback
        ctx.fillText(`Feedback: ${feedback}`, 10, 60); // Adjusted padding
    }

    // Initialize and start the detection
    await poseDetection.init();
}

// Start the pose detection once the page loads
window.onload = () => {
    startPoseDetection();
};
