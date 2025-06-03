// Enhanced apiService.js - Support for Multiple Models per Camera
import axios from 'axios';

const API_BASE_URL = 'http://localhost:8001';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
  headers: { 'Content-Type': 'application/json' },
});

// ✅ Fetch available cameras
export const getCameras = async () => {
  try {
    const response = await api.get('/api/cameras');
    return response.data;
  } catch (error) {
    console.error('Failed to fetch cameras:', error);
    throw error;
  }
};

// ✅ Enhanced Start Tracking with Multi-Model Support
export const startTracking = async (cameraId, model, options = {}) => {
  try {
    const requestBody = {
      model,
      ...options // Include pairId, scenario, enableMultiModel, etc.
    };
    
    console.log(`Starting tracking for camera ${cameraId} with options:`, requestBody);
    
    // Use a different endpoint for multi-model support if available
    const endpoint = options.enableMultiModel 
      ? `/api/cameras/${cameraId}/start_tracking_multimodel`
      : `/api/cameras/${cameraId}/start_tracking`;
    
    const response = await api.post(endpoint, requestBody);
    return response.data;
  } catch (error) {
    console.error(`Failed to start tracking for camera ${cameraId}:`, error);
    
    // If multi-model endpoint doesn't exist, fallback to standard endpoint
    if (error.response?.status === 404 && options.enableMultiModel) {
      console.log('Multi-model endpoint not available, using standard endpoint');
      return await startTrackingFallback(cameraId, model, options);
    }
    
    throw error;
  }
};

// Fallback method for backends that don't support multi-model yet
const startTrackingFallback = async (cameraId, model, options = {}) => {
  try {
    // For fallback, we'll start tracking with a unique session ID
    const sessionId = options.pairId || `${cameraId}_${model}_${Date.now()}`;
    
    const requestBody = {
      model,
      session_id: sessionId,
      scenario: options.scenario || 'multi-object'
    };
    
    console.log(`Fallback: Starting tracking for camera ${cameraId} session ${sessionId}`);
    
    const response = await api.post(`/api/cameras/${cameraId}/start_tracking`, requestBody);
    return response.data;
  } catch (error) {
    console.error(`Fallback tracking failed for camera ${cameraId}:`, error);
    throw error;
  }
};

// ✅ Enhanced Stop Tracking with Multi-Model Support
export const stopTracking = async (cameraId, options = {}) => {
  try {
    const requestBody = options.model ? {
      model: options.model,
      pair_id: options.pairId
    } : {};
    
    console.log(`Stopping tracking for camera ${cameraId} with options:`, requestBody);
    
    // Use specific endpoint for multi-model if options provided
    const endpoint = options.model 
      ? `/api/cameras/${cameraId}/stop_tracking_model`
      : `/api/cameras/${cameraId}/stop_tracking`;
    
    const response = await api.post(endpoint, requestBody);
    return response.data;
  } catch (error) {
    console.error(`Failed to stop tracking for camera ${cameraId}:`, error);
    
    // Fallback to standard stop endpoint
    if (error.response?.status === 404) {
      console.log('Model-specific stop endpoint not available, using standard endpoint');
      const response = await api.post(`/api/cameras/${cameraId}/stop_tracking`);
      return response.data;
    }
    
    throw error;
  }
};

// ✅ Get Detection Status with Model Information
export const getDetectionStatus = async (cameraId, model = null) => {
  try {
    const url = model 
      ? `/api/cameras/${cameraId}/detection_status?model=${model}`
      : `/api/cameras/${cameraId}/detection_status`;
      
    const response = await api.get(url);
    return response.data;
  } catch (error) {
    console.error(`Failed to get detection status for camera ${cameraId}:`, error);
    throw error;
  }
};

// ✅ Get Multi-Model Status (if backend supports it)
export const getMultiModelStatus = async (cameraId) => {
  try {
    const response = await api.get(`/api/cameras/${cameraId}/multimodel_status`);
    return response.data;
  } catch (error) {
    console.error(`Failed to get multi-model status for camera ${cameraId}:`, error);
    // Return empty status if endpoint doesn't exist
    if (error.response?.status === 404) {
      return { models: [], active_models: [] };
    }
    throw error;
  }
};

// ✅ Health Check with Multi-Model Info
export const getSystemHealth = async () => {
  try {
    const response = await api.get('/api/health');
    return response.data;
  } catch (error) {
    console.error('Failed to get system health:', error);
    throw error;
  }
};

// ✅ Get Available Models
export const getAvailableModels = async () => {
  try {
    const response = await api.get('/api/models');
    return response.data;
  } catch (error) {
    console.error('Failed to fetch available models:', error);
    // Return default models if endpoint doesn't exist
    if (error.response?.status === 404) {
      return [
        { id: 'yolov8n', name: 'YOLOv8 Nano', description: 'Fastest, lowest accuracy' },
        { id: 'yolov8s', name: 'YOLOv8 Small', description: 'Balanced speed and accuracy' },
        { id: 'yolov8m', name: 'YOLOv8 Medium', description: 'Good accuracy, moderate speed' },
        { id: 'yolov8l', name: 'YOLOv8 Large', description: 'High accuracy, slower' }
      ];
    }
    throw error;
  }
};

// ✅ Test Multi-Model Capability
export const testMultiModelSupport = async () => {
  try {
    const response = await api.get('/api/capabilities/multimodel');
    return response.data?.supported || false;
  } catch (error) {
    // If endpoint doesn't exist, assume no multi-model support
    return false;
  }
};

// ✅ Get Camera Stream URL with Model Support
export const getCameraStreamUrl = (cameraId, options = {}) => {
  const baseUrl = `${API_BASE_URL}/api/cameras/${cameraId}`;
  const params = new URLSearchParams();
  
  if (options.model) params.append('model', options.model);
  if (options.pairId) params.append('pair_id', options.pairId);
  if (options.session) params.append('session', options.session);
  params.append('t', Date.now()); // Cache buster
  
  const streamType = options.detection ? 'detection_stream' : 'stream';
  return `${baseUrl}/${streamType}?${params.toString()}`;
};

// ✅ Helper: Get Detection Stream URL for specific model
export const getDetectionStreamUrl = (cameraId, model, pairId = null) => {
  return getCameraStreamUrl(cameraId, {
    model,
    pairId,
    detection: true
  });
};

// ✅ Helper: Get Regular Stream URL
export const getRegularStreamUrl = (cameraId, pairId = null) => {
  return getCameraStreamUrl(cameraId, {
    pairId,
    detection: false
  });
};