// --------------- src/services/cameraService.js ---------------
// Camera connection and management service
const connectToCamera = async (camera) => {
  try {
    // For local camera access
    if (camera.type === 'local') {
      const constraints = {
        video: {
          deviceId: camera.deviceId ? { exact: camera.deviceId } : undefined,
          width: { ideal: 1280 },
          height: { ideal: 720 }
        }
      };
      
      return await navigator.mediaDevices.getUserMedia(constraints);
    }
    
    // For IP cameras or RTSP streams, we would use a backend proxy
    // In a real app, you'd send a request to your backend which would handle the stream
    if (camera.type === 'ip' || camera.type === 'rtsp') {
      // This is a placeholder for actual implementation
      console.log(`Connecting to IP/RTSP camera: ${camera.url}`);
      
      // In a real app, we would either:
      // 1. Use WebRTC to receive the stream from a backend proxy
      // 2. Use WebSockets to receive frames as JPEGs or base64
      
      // For now, we'll mock it with a placeholder
      const mockStream = await navigator.mediaDevices.getUserMedia({ video: true });
      return mockStream;
    }
    
    throw new Error(`Unsupported camera type: ${camera.type}`);
  } catch (error) {
    console.error(`Failed to connect to camera:`, error);
    throw error;
  }
};

const disconnectFromCamera = (stream) => {
  if (stream) {
    stream.getTracks().forEach(track => track.stop());
  }
};

export { connectToCamera, disconnectFromCamera };

// --------------- src/services/detectionService.js ---------------
// Object detection service
const activeDetections = {};

const startObjectDetection = (cameraId, modelName, callback) => {
  // In a real app, we would connect to a backend that runs the YOLO model
  console.log(`Starting object detection on camera ${cameraId} with model ${modelName}`);
  
  // Mock detection results with a timer
  const timer = setInterval(() => {
    // Simulate object detection results
    const mockDetections = [
      {
        class: 'person',
        confidence: Math.random() * 0.3 + 0.7, // 0.7 to 1.0
        bbox: [Math.random() * 200, Math.random() * 150, 100, 200]
      },
      {
        class: 'car',
        confidence: Math.random() * 0.5 + 0.3, // 0.3 to 0.8
        bbox: [Math.random() * 300 + 100, Math.random() * 200 + 50, 150, 100]
      }
    ];
    
    // Only include some detections sometimes to make it more realistic
    const filteredDetections = mockDetections.filter(() => Math.random() > 0.3);
    
    callback({ detections: filteredDetections });
  }, 1000);
  
  // Store the timer to clean up later
  activeDetections[cameraId] = timer;
};

const stopObjectDetection = (cameraId) => {
  if (activeDetections[cameraId]) {
    clearInterval(activeDetections[cameraId]);
    delete activeDetections[cameraId];
    console.log(`Stopped object detection on camera ${cameraId}`);
  }
};

export { startObjectDetection, stopObjectDetection };

// --------------- src/services/recordingService.js ---------------
// Video recording service
const activeRecordings = {};

const startRecording = (cameraId, outputDir) => {
  console.log(`Started recording for camera ${cameraId} to ${outputDir}`);
  
  // In a real application, we would:
  // 1. Use MediaRecorder API for local browser recording
  // 2. Or send a command to the backend to start recording there
  
  // For this demo, we'll just set a flag
  activeRecordings[cameraId] = {
    startTime: new Date(),
    outputDir
  };
  
  return true;
};

const stopRecording = (cameraId) => {
  if (activeRecordings[cameraId]) {
    const { startTime, outputDir } = activeRecordings[cameraId];
    const duration = (new Date() - startTime) / 1000; // in seconds
    
    console.log(`Stopped recording for camera ${cameraId}. Duration: ${duration}s. Saved to ${outputDir}`);
    
    delete activeRecordings[cameraId];
    return { success: true, duration, outputDir };
  }
  
  return { success: false, error: 'No active recording found' };
};

export { startRecording, stopRecording };
