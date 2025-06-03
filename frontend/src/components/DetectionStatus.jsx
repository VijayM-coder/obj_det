// components/DetectionStatus.jsx
import React, { useState, useEffect } from 'react';
import { detectionMonitor } from '../services/detectionMonitor';
import './DetectionStatus.css';

const DetectionStatus = ({ isTrackingActive, trackingPairs }) => {
  const [detectionStats, setDetectionStats] = useState({});
  const [systemHealth, setSystemHealth] = useState(null);
  const [isMonitoring, setIsMonitoring] = useState(false);

  useEffect(() => {
    // Start/stop monitoring based on tracking state
    if (isTrackingActive && trackingPairs.length > 0) {
      const cameraIds = trackingPairs.map(pair => pair.cameraId);
      detectionMonitor.startMonitoring(cameraIds);
      setIsMonitoring(true);
    } else {
      detectionMonitor.stopMonitoring();
      setIsMonitoring(false);
      setDetectionStats({});
      setSystemHealth(null);
    }
  }, [isTrackingActive, trackingPairs]);

  useEffect(() => {
    // Subscribe to detection monitor events
    const unsubscribe = detectionMonitor.addListener((event, data) => {
      switch (event) {
        case 'monitoring_started':
          console.log('Detection monitoring started:', data);
          setIsMonitoring(true);
          break;
          
        case 'monitoring_stopped':
          console.log('Detection monitoring stopped');
          setIsMonitoring(false);
          setDetectionStats({});
          setSystemHealth(null);
          break;
          
        case 'status_updated':
          setDetectionStats(prev => ({
            ...prev,
            [data.cameraId]: data.status
          }));
          break;
          
        case 'health_updated':
          setSystemHealth(data);
          break;
          
        case 'detection_recorded':
          // Update local stats immediately
          setDetectionStats(prev => ({
            ...prev,
            [data.cameraId]: {
              ...prev[data.cameraId],
              total_detections: data.stats.totalDetections,
              last_detection: data.detection
            }
          }));
          break;
          
        case 'health_error':
          console.error('Detection health check failed:', data.error);
          break;
          
        default:
          console.log('Unknown detection monitor event:', event, data);
      }
    });

    return () => {
      unsubscribe();
    };
  }, []);

  // Get status indicator class
  const getStatusClass = (status) => {
    switch (status) {
      case 'connected': return 'status-connected';
      case 'connecting': return 'status-connecting';
      case 'error': return 'status-error';
      case 'closed': return 'status-closed';
      default: return 'status-unknown';
    }
  };

  // Get confidence color
  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.8) return '#ff4444';
    if (confidence >= 0.6) return '#ff8800';
    return '#44aa44';
  };

  // Format timestamp
  const formatTime = (timestamp) => {
    if (!timestamp) return 'Never';
    try {
      return new Date(timestamp).toLocaleTimeString();
    } catch {
      return 'Invalid';
    }
  };

  // Calculate total detections
  const totalDetections = Object.values(detectionStats).reduce(
    (sum, stats) => sum + (stats.total_detections || 0), 0
  );

  if (!isTrackingActive) {
    return (
      <div className="detection-status inactive">
        <div className="status-header">
          <h3>Detection System</h3>
          <span className="status-badge inactive">Inactive</span>
        </div>
        <p className="status-message">Start tracking to activate detection monitoring</p>
      </div>
    );
  }

  return (
    <div className="detection-status active">
      <div className="status-header">
        <h3>Detection System</h3>
        <span className="status-badge active">
          {isMonitoring ? 'Monitoring' : 'Starting...'}
        </span>
      </div>

      {/* System Health Overview */}
      {systemHealth && (
        <div className="system-health">
          <div className="health-stats">
            <div className="health-item">
              <span className="health-label">Model</span>
              <span className="health-value">
                {systemHealth.current_model || 'None'}
              </span>
            </div>
            <div className="health-item">
              <span className="health-label">Active Cameras</span>
              <span className="health-value">
                {systemHealth.active_cameras}/{systemHealth.total_cameras}
              </span>
            </div>
            <div className="health-item">
              <span className="health-label">Total Detections</span>
              <span className="health-value">{totalDetections}</span>
            </div>
          </div>
        </div>
      )}

      {/* Camera Status Details */}
      <div className="cameras-status">
        <h4>Camera Status</h4>
        <div className="camera-list">
          {trackingPairs.map(pair => {
            const stats = detectionStats[pair.cameraId] || {};
            const isActive = stats.isActive;
            const lastDetection = stats.last_detection;
            
            return (
              <div key={pair.cameraId} className="camera-status-item">
                <div className="camera-info">
                  <div className="camera-header">
                    <span className="camera-name">Camera {pair.cameraId}</span>
                    <div className={`status-indicator ${getStatusClass(stats.connection_status)}`}>
                      <span className="status-dot"></span>
                      <span className="status-text">
                        {stats.connection_status || 'Unknown'}
                      </span>
                    </div>
                  </div>
                  
                  <div className="camera-details">
                    <div className="detail-row">
                      <span className="detail-label">Model:</span>
                      <span className="detail-value">{pair.model}</span>
                    </div>
                    
                    <div className="detail-row">
                      <span className="detail-label">Detections:</span>
                      <span className="detail-value">
                        {stats.total_detections || 0}
                        {stats.recent_detections > 0 && (
                          <span className="recent-count">
                            ({stats.recent_detections} recent)
                          </span>
                        )}
                      </span>
                    </div>
                    
                    <div className="detail-row">
                      <span className="detail-label">Last Detection:</span>
                      <span className="detail-value">
                        {lastDetection ? (
                          <span>
                            {lastDetection.class_name} at {formatTime(lastDetection.timestamp)}
                            <span 
                              className="confidence-badge"
                              style={{ backgroundColor: getConfidenceColor(lastDetection.confidence) }}
                            >
                              {Math.round(lastDetection.confidence * 100)}%
                            </span>
                          </span>
                        ) : (
                          'None'
                        )}
                      </span>
                    </div>
                    
                    <div className="detail-row">
                      <span className="detail-label">WebSocket:</span>
                      <span className="detail-value">
                        {stats.websocket_clients || 0} connection(s)
                      </span>
                    </div>
                  </div>
                </div>
                
                {!isActive && (
                  <div className="camera-warning">
                    ⚠️ Detection not active for this camera
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* Performance Metrics */}
      <div className="performance-metrics">
        <h4>Performance</h4>
        <div className="metrics-grid">
          <div className="metric-item">
            <span className="metric-label">Avg. Confidence</span>
            <span className="metric-value">
              {Object.values(detectionStats).length > 0 ? (
                Math.round(
                  Object.values(detectionStats)
                    .filter(s => s.last_detection)
                    .reduce((sum, s) => sum + s.last_detection.confidence, 0) /
                  Object.values(detectionStats).filter(s => s.last_detection).length * 100
                ) + '%'
              ) : (
                'N/A'
              )}
            </span>
          </div>
          
          <div className="metric-item">
            <span className="metric-label">Detection Rate</span>
            <span className="metric-value">
              {Object.values(detectionStats).reduce((sum, s) => sum + (s.recent_detections || 0), 0)} / min
            </span>
          </div>
          
          <div className="metric-item">
            <span className="metric-label">System Status</span>
            <span className={`metric-value ${systemHealth?.system_status === 'healthy' ? 'healthy' : 'warning'}`}>
              {systemHealth?.system_status === 'healthy' ? '✅ Healthy' : '⚠️ Issues'}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DetectionStatus;