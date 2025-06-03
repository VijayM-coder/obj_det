// services/detectionMonitor.js - Clean detection monitoring service
class DetectionMonitor {
  constructor() {
    this.isMonitoring = false;
    this.detectionStats = new Map(); // camera_id -> stats
    this.statusCheckInterval = null;
    this.healthCheckInterval = null;
    this.listeners = new Set();
  }

  // Start monitoring detections when tracking begins
  startMonitoring(cameraIds) {
    console.log('🔍 Starting detection monitoring for cameras:', cameraIds);
    this.isMonitoring = true;
    
    // Initialize stats for each camera
    cameraIds.forEach(cameraId => {
      this.detectionStats.set(cameraId, {
        totalDetections: 0,
        lastDetection: null,
        confidence: 0,
        isActive: false,
        connectionStatus: 'connecting'
      });
    });

    // Start periodic status checks
    this.startStatusChecks();
    this.startHealthChecks();
    
    // Notify listeners
    this.notifyListeners('monitoring_started', { cameraIds });
  }

  // Stop monitoring when tracking stops
  stopMonitoring() {
    console.log('⏹️ Stopping detection monitoring');
    this.isMonitoring = false;
    
    // Clear intervals
    if (this.statusCheckInterval) {
      clearInterval(this.statusCheckInterval);
      this.statusCheckInterval = null;
    }
    
    if (this.healthCheckInterval) {
      clearInterval(this.healthCheckInterval);
      this.healthCheckInterval = null;
    }

    // Clear stats
    this.detectionStats.clear();
    
    // Notify listeners
    this.notifyListeners('monitoring_stopped');
  }

  // Check detection status for all cameras
  startStatusChecks() {
    this.statusCheckInterval = setInterval(async () => {
      if (!this.isMonitoring) return;

      for (const cameraId of this.detectionStats.keys()) {
        try {
          const response = await fetch(`http://localhost:8001/api/cameras/${cameraId}/detection_status`);
          
          if (response.ok) {
            const status = await response.json();
            
            // Update stats
            const currentStats = this.detectionStats.get(cameraId) || {};
            this.detectionStats.set(cameraId, {
              ...currentStats,
              totalDetections: status.total_detections,
              recentDetections: status.recent_detections,
              lastDetection: status.last_detection,
              isActive: status.detection_enabled && status.camera_status === 'connected',
              connectionStatus: status.camera_status,
              modelLoaded: status.model_loaded,
              currentModel: status.current_model,
              websocketClients: status.websocket_clients
            });

            // Notify listeners of status update
            this.notifyListeners('status_updated', { cameraId, status });
          }
        } catch (error) {
          console.error(`Failed to check status for camera ${cameraId}:`, error);
          
          // Update with error status
          const currentStats = this.detectionStats.get(cameraId) || {};
          this.detectionStats.set(cameraId, {
            ...currentStats,
            connectionStatus: 'error',
            isActive: false
          });
        }
      }
    }, 5000); // Check every 5 seconds
  }

  // Check overall system health
  startHealthChecks() {
    this.healthCheckInterval = setInterval(async () => {
      if (!this.isMonitoring) return;

      try {
        const response = await fetch('http://localhost:8001/api/detection/health');
        
        if (response.ok) {
          const health = await response.json();
          this.notifyListeners('health_updated', health);
        }
      } catch (error) {
        console.error('Health check failed:', error);
        this.notifyListeners('health_error', { error: error.message });
      }
    }, 10000); // Check every 10 seconds
  }

  // Record a detection event
  recordDetection(cameraId, detection) {
    if (!this.detectionStats.has(cameraId)) return;

    const stats = this.detectionStats.get(cameraId);
    stats.totalDetections += 1;
    stats.lastDetection = detection;
    stats.confidence = detection.confidence;
    
    this.detectionStats.set(cameraId, stats);
    
    // Notify listeners
    this.notifyListeners('detection_recorded', { cameraId, detection, stats });
  }

  // Get stats for a camera
  getStats(cameraId) {
    return this.detectionStats.get(cameraId) || null;
  }

  // Get all stats
  getAllStats() {
    return Object.fromEntries(this.detectionStats);
  }

  // Add listener for detection events
  addListener(callback) {
    this.listeners.add(callback);
    return () => this.listeners.delete(callback);
  }

  // Notify all listeners
  notifyListeners(event, data) {
    this.listeners.forEach(callback => {
      try {
        callback(event, data);
      } catch (error) {
        console.error('Error in detection monitor listener:', error);
      }
    });
  }

  // Check if monitoring is active
  isActive() {
    return this.isMonitoring;
  }
}

// Create singleton instance
export const detectionMonitor = new DetectionMonitor();