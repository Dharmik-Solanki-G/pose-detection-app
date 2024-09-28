import Detection from "./detection.js";
import poseDetectionConfig from "./poseDetectionConfig.js";
import {
    PoseLandmarker,
    DrawingUtils
} from "https://cdn.skypack.dev/@mediapipe/tasks-vision@0.10.0";

class PoseDetection extends Detection {
    constructor(stream) {
        super(poseDetectionConfig , PoseLandmarker)
        this.setStream(stream)
    }

    async onDetection() {
        if(!this.landMarker) return
        this.canvas.width = this.video.videoWidth;
        this.canvas.height = this.video.videoHeight;

        // Now let's start detecting the stream.
        if (this.runningMode === "IMAGE") {
            this.runningMode = "VIDEO";
            await this.landMarker.setOptions({
                runningMode: "VIDEO"
            });
        }
        let startTimeMs = performance.now();
        if (this.lastVideoTime !== this.video.currentTime) {
            this.lastVideoTime = this.video.currentTime;
            this.landMarker?.detectForVideo(this.video, startTimeMs, (result) => {

                if (Array.isArray(result.landmarks) && result.landmarks.length > 0) {
                    if (this.resultCallback && result) {
                        this.resultCallback(result, this.canvas.width, this.canvas.height);
                    }
                }

                let pointerColor = new Array(32).fill("#FF0000");
                for(let i = 0 ; i < 32 ; i++){
                    if(this.correctPoints && this.correctPoints[i]){
                        pointerColor[i] = "#00FF00";
                    }
                }

                let connectionColor = new Array(35).fill("#FF0000");

                if(this.correctPoints){
                    for (let i = 0; i < this.connections.length; i++) {
                        const start_idx =this.connections[i][0];
                        const end_idx = this.connections[i][1];
                        const start_point = this.correctPoints[start_idx];
                        const end_point = this.correctPoints[end_idx];
                        if (start_point && end_point) {
                            connectionColor[i] = "#00FF00";
                        }
                    }
                }

                this.canvasCtx.save();
                this.canvasCtx.clearRect(0, 0, this.canvas.width, this.canvas.height);
                for (const landmark of result.landmarks) {
                    for(let i =0 ; i<32; i++){
                         drawLandmarks(this.canvasCtx, [landmark[i]], {
                                color: pointerColor[i], // Cycle through colors
                                radius: (data) => DrawingUtils.lerp(data.from.z, -0.15, 0.1, 5, 1),
                            });
                    }

                    this.LandMarker.POSE_CONNECTIONS.forEach((connection, index) => {
                        this.drawingUtils.drawConnectors(landmark, [connection], {
                            color: connectionColor[index],
                            lineWidth: 5
                        });
                    });
                    
                }
            });
        }

        this.canvasCtx.restore();

        // // Call this function again to keep predicting when the browser is ready.
        if (this.webcamRunning === true) {
            setTimeout(() => {
                window.requestAnimationFrame(this.onDetection.bind(this));
            }, 50);
        }

    }
}

export default PoseDetection