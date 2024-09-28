const poseDetectionConfig = {
    connections: [
        [0, 1], [1, 2], [2, 3], [3, 7],
        [0, 4], [4, 5], [5, 6], [6, 8],
        [9,10], [11,12], [11,13], [13,15],
        [15,17], [15,19], [15,21], [17,19],
        [12,14], [14,16], [16,18], [16,20],
        [16,22], [18,20], [11,23], [12,24],
        [23,24], [23,25], [24,26], [25,27],
        [26,28], [27,29], [28,30], [29,31],
        [30,32], [27,31], [28,32]
    ],
    visionPath: "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm",
    baseOptions: {
        modelAssetPath: `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task`,
        delegate: "GPU"
    },
    numPoses: 1
}

export default poseDetectionConfig