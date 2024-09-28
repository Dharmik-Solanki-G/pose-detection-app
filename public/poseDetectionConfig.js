const poseDetectionConfig = {
    connections: [
        // Add connections for the nose
        [0, 1],   // Nose to left shoulder
        [1, 2],   // Left shoulder to left elbow
        [2, 3],   // Left elbow to left wrist
        [3, 7],   // Left wrist to left hip
        [0, 4],   // Nose to right shoulder
        [4, 5],   // Right shoulder to right elbow
        [5, 6],   // Right elbow to right wrist
        [6, 8],   // Right wrist to right hip
        [9, 10],  // Left hip to left knee
        [11, 12], // Right hip to right knee
        [11, 13], // Right knee to right ankle
        [13, 15], // Right ankle to right foot index
        [15, 17], // Right foot index to right foot pinky
        [15, 19], // Right foot index to right foot ring
        [15, 21], // Right foot index to right foot middle
        [17, 19], // Right foot pinky to right foot ring
        [12, 14], // Left knee to left ankle
        [14, 16], // Left ankle to left foot index
        [16, 18], // Left foot index to left foot pinky
        [16, 20], // Left foot index to left foot ring
        [16, 22], // Left foot index to left foot middle
        [18, 20], // Left foot pinky to left foot ring
        [11, 23], // Left hip to spine base
        [12, 24], // Right hip to spine base
        [23, 24], // Spine base to spine mid
        [23, 25], // Spine base to neck
        [24, 26], // Spine mid to neck
        [25, 27], // Neck to head
        [26, 28], // Spine mid to left shoulder
        [27, 29], // Head to left shoulder
        [28, 30], // Left shoulder to right shoulder
        [29, 31], // Head to right shoulder
        [30, 32], // Left shoulder to right shoulder
        [27, 31], // Neck to right shoulder
        [28, 32]  // Left shoulder to right shoulder
    ],
    visionPath: "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm",
    baseOptions: {
        modelAssetPath: `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task`,
        delegate: "GPU"
    },
    numPoses: 1
}

export default poseDetectionConfig;
