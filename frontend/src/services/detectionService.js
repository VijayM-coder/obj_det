// src/services/detectionService.js

// Start object detection for a specific camera
export const startObjectDetection = async (cameraId, modelId) => {
  try {
    const response = await fetch(`http://localhost:8001/api/cameras/${cameraId}/start_tracking`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        model: modelId
      })
    });
    
    if (!response.ok) {
      throw new Error(`Failed to start object detection: ${response.statusText}`);
    }
    
    const data = await response.json();
    console.log('Object detection started:', data);
    return data;
  } catch (error) {
    console.error('Error starting object detection:', error);
    throw error;
  }
};

// Stop object detection for a specific camera
export const stopObjectDetection = async (cameraId) => {
  try {
    const response = await fetch(`http://localhost:8001/api/cameras/${cameraId}/stop_tracking`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      }
    });
    
    if (!response.ok) {
      throw new Error(`Failed to stop object detection: ${response.statusText}`);
    }
    
    const data = await response.json();
    console.log('Object detection stopped:', data);
    return data;
  } catch (error) {
    console.error('Error stopping object detection:', error);
    throw error;
  }
};

// Get latest detections for a camera
export const getDetections = async (cameraId) => {
  try {
    const response = await fetch(`http://localhost:8001/api/cameras/${cameraId}/detections`, {
      method: 'GET'
    });
    
    if (!response.ok) {
      throw new Error(`Failed to get detections: ${response.statusText}`);
    }
    
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error getting detections:', error);
    return [];
  }
};