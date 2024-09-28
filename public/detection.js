import {
    DrawingUtils,
    FilesetResolver,
} from "https://cdn.skypack.dev/@mediapipe/tasks-vision@0.10.0";

class Detection {
    constructor(config, Landmarker) {
        this.LandMarker = Landmarker
        this.config = config
        this.landMarker = null;
        this.canvas = null;
        this.ctx = null;
        this.correctPoints = null
        this.correctPoints2 = null
        this.lastVideoTime = -1;
        this.runningMode = "Image";
        this.webcamRunning = false;
        this.results = undefined;

        this.setConnections(config.connections)
        this.setVideo()
        this.setVideoContainer()
        this.createCanvas()
    }
    async init() {
        try {
            await this.createLandMarker()
            this.enableCam()
        } catch (error) {
            console.error("Failed to initialize hand detection:", error);
        }
    }
    async createLandMarker() {
        try {
            const vision = await FilesetResolver.forVisionTasks(this.config.visionPath);
            this.landMarker = await this.LandMarker.createFromOptions(vision, {
                baseOptions: {
                    modelAssetPath: this.config.baseOptions.modelAssetPath,
                    delegate: this.config.baseOptions.delegate
                },
                runningMode: this.runningMode,
                numHands: this.config?.numHands,
                numPoses: this.config?.numPoses,
            });

        } catch (error) {
            console.error("Failed to initialize Detection:", error);
        }

    }
    async getLandMarker() {
        return this.landMarker
    }
    async removeLandMarker() {
        if (this.landMarker) {
            await this.landMarker.close(); // Properly dispose of the landmarker
            this.landMarker = null;
        }
    }
    async setCorrectPoints(correctPoints, correctPoints2) {
        this.correctPoints = correctPoints;
        this.correctPoints2 = correctPoints2;
    }
    async getCorrectPoints() {
        return this.correctPoints
    }
    async getCorrectPoints2() {
        return this.correctPoints2
    }
    async setStream(stream) {
        this.stream = stream
    }
    getStream() {
        return this.stream
    }
    async setConnections(connections) {
        this.connections = connections
    }
    async getConnections() {
        return this.connections
    }
    async setResults(results) {
        this.results = results
    }
    async getResults() {
        return this.results
    }
    async setResultCallback(callback) {
        this.resultCallback = callback
    }
    async getResultCallback() {
        return this.resultCallback
    }
    async createCanvas() {
        this.canvas = document.getElementById("local-canva");
        if (!this.canvas) {
            this.canvas = document.createElement("canvas");
            this.canvas.id = "local-canva";

            let videoContainer = await this.getVideoContainer()
            videoContainer.appendChild(this.canvas);
        }
        this.canvasCtx = this.canvas.getContext("2d");

        this.drawingUtils = new DrawingUtils(this.canvasCtx);
    }
    async clearCanvas() {
        if (this.canvasCtx && this.canvas) {
            this.canvasCtx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        }
    }
    async removeCanvas() {
        this.canvas.remove()
    }
    async setVideo() {
        this.video = document.querySelector(".local-video")
    }
    getVideo() {
        return this.video
    }
    async setVideoContainer() {
        this.videoContainer = document.querySelector(".local-video-container");
    }
    async getVideoContainer() {
        return this.videoContainer
    }

    async setWebCamStatus(status) {
        this.webcamRunning = status
    }

    async enableCam() {
        if (!this.landMarker) {
            console.log("Wait! Landmarker not loaded yet.");
            return;
        }
        this.setWebCamStatus(true)

        let video = this.getVideo()
        video.srcObject = this.getStream()

        video.addEventListener("loadeddata", this.onLoad)
    }
    onLoad = async () => {
        try {
            this.onDetection()
        } catch (error) {
            this.stopDetection()
        }
    }
    async onDetection() { }
    async stopDetection() {
        this.webcamRunning = false;
        this.canvasCtx?.clearRect(0, 0, this.canvas.width, this.canvas.height);
        if (this.video) {
            this.video.removeEventListener("loadeddata", this.onLoadedDataHandler);
        }
        if (this.landMarker) {
            this.landMarker.close();
            this.landMarker = undefined;
        }
    }
    async restartDetection(resultCallback, newStream) {
        this.setResultCallback(resultCallback)
        this.stopDetection()

        if (newStream) {
            this.setStream(newStream)
        }

        await this.init()
    }

}

export default Detection