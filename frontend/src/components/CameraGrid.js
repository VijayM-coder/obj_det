// Fixed CameraGrid.js - Support Multiple Camera Instances with Different Models
import React, { useEffect, useRef, useState } from 'react';
import './CameraGrid.css';

function CameraGrid({ 
  cameraModelPairs, 
  trackingPairs = [], 
  gridLayout, 
  alertedCameras = new Set(), 
  flashEnabled = true,
  isTrackingActive = false 
}) {
  const imgRefs = useRef({});
  const streamErrorTimers = useRef({});
  const [streamStatus, setStreamStatus] = useState({});
  const [cameraStats, setCameraStats] = useState({});
  const { rows, cols } = gridLayout;

  // Clean up any pending timers when component unmounts
  useEffect(() => {
    return () => {
      Object.values(streamErrorTimers.current).forEach(timer => {
        if (timer) clearTimeout(timer);
      });
    };
  }, []);

  useEffect(() => {
    // Clear existing error timers when dependencies change
    Object.values(streamErrorTimers.current).forEach(timer => {
      if (timer) clearTimeout(timer);
    });

    console.log('CameraGrid: Setting up streams for', cameraModelPairs.length, 'camera configurations');

    // Initialize stream connections for each camera-model configuration
    cameraModelPairs.forEach((pair, index) => {
      const camera = pair.camera;
      const pairId = pair.pairId || `${camera.id}_${pair.model}_${index}`;
      const imgElement = imgRefs.current[pairId];
      
      if (imgElement) {
        // Find if this specific camera-model pair is in tracking mode
        const isTracking = trackingPairs.some(tp => 
          tp.cameraId === camera.id && tp.model === pair.model
        );
        
        const model = isTracking 
          ? trackingPairs.find(tp => tp.cameraId === camera.id && tp.model === pair.model)?.model 
          : pair.model;
        
        // Set loading state for this specific pair
        setStreamStatus(prev => ({
          ...prev,
          [pairId]: { loading: true, error: false, lastUpdate: Date.now() }
        }));
        
        // Add timestamp to prevent caching
        const timestamp = Date.now();
        
        // Set appropriate stream URL - use unique URL for each model
        const streamUrl = isTracking
          ? `http://localhost:8001/api/cameras/${camera.id}/detection_stream?model=${model}&pair_id=${pairId}&session=${pairId}&t=${timestamp}`
          : `http://localhost:8001/api/cameras/${camera.id}/stream?pair_id=${pairId}&session=${pairId}&t=${timestamp}`;
        
        console.log(`Setting stream URL for camera ${camera.id} (${model}):`, streamUrl);
        imgElement.src = streamUrl;
        
        // Handle successful load
        imgElement.onload = () => {
          console.log(`Stream loaded successfully for camera ${camera.id} with model ${model}`);
          setStreamStatus(prev => ({
            ...prev,
            [pairId]: { 
              loading: false, 
              error: false, 
              lastUpdate: Date.now(),
              connected: true 
            }
          }));
        };
        
        // Handle stream errors with exponential backoff
        imgElement.onerror = (error) => {
          console.warn(`Stream error for camera ${camera.id} (${model}):`, error);
          setStreamStatus(prev => ({
            ...prev, 
            [pairId]: { 
              loading: false, 
              error: true, 
              lastUpdate: Date.now(),
              connected: false 
            }
          }));
          
          // Clear any existing timer for this pair
          if (streamErrorTimers.current[pairId]) {
            clearTimeout(streamErrorTimers.current[pairId]);
          }
          
          // Set new reconnection timer with exponential backoff
          const retryDelay = Math.min(2000 * Math.pow(2, (prev?.[pairId]?.retryCount || 0)), 30000);
          
          streamErrorTimers.current[pairId] = setTimeout(() => {
            // Only attempt reconnection if component is still mounted
            if (imgElement && imgRefs.current[pairId]) {
              const newTimestamp = Date.now();
              console.log(`Retrying connection for camera ${camera.id} (${model})`);
              
              setStreamStatus(prev => ({
                ...prev,
                [pairId]: { 
                  loading: true, 
                  error: false, 
                  retryCount: (prev?.[pairId]?.retryCount || 0) + 1,
                  lastUpdate: Date.now()
                }
              }));
              
              const retryUrl = isTracking
                ? `http://localhost:8001/api/cameras/${camera.id}/detection_stream?model=${model}&pair_id=${pairId}&session=${pairId}&t=${newTimestamp}`
                : `http://localhost:8001/api/cameras/${camera.id}/stream?pair_id=${pairId}&session=${pairId}&t=${newTimestamp}`;
              
              imgElement.src = retryUrl;
            }
          }, retryDelay);
        };
      }
    });

    // Cleanup function to stop all streams when component updates or unmounts
    return () => {
      cameraModelPairs.forEach((pair, index) => {
        const pairId = pair.pairId || `${pair.camera.id}_${pair.model}_${index}`;
        const imgElement = imgRefs.current[pairId];
        if (imgElement) {
          imgElement.onload = null;
          imgElement.onerror = null;
          imgElement.src = '';
        }
        
        // Clear any pending reconnection timers
        if (streamErrorTimers.current[pairId]) {
          clearTimeout(streamErrorTimers.current[pairId]);
          delete streamErrorTimers.current[pairId];
        }
      });
    };
  }, [cameraModelPairs, trackingPairs]);

  // Periodically fetch camera statistics when tracking is active
  useEffect(() => {
    if (!isTrackingActive) {
      setCameraStats({});
      return;
    }

    const fetchStats = async () => {
      const stats = {};
      
      // Fetch stats for each unique camera (not each pair)
      const uniqueCameras = [...new Set(cameraModelPairs.map(p => p.camera.id))];
      
      for (const cameraId of uniqueCameras) {
        try {
          const response = await fetch(`http://localhost:8001/api/cameras/${cameraId}/detection_status`);
          if (response.ok) {
            const data = await response.json();
            stats[cameraId] = {
              totalDetections: data.total_detections || 0,
              recentDetections: data.recent_detections || 0,
              lastDetection: data.last_detection,
              detectionEnabled: data.detection_enabled,
              cameraStatus: data.camera_status
            };
          }
        } catch (error) {
          console.warn(`Failed to fetch stats for camera ${cameraId}:`, error);
        }
      }
      
      setCameraStats(stats);
    };

    // Initial fetch
    fetchStats();
    
    // Set up periodic updates
    const statsInterval = setInterval(fetchStats, 5000); // Every 5 seconds
    
    return () => clearInterval(statsInterval);
  }, [isTrackingActive, cameraModelPairs]);

  // Get model name for display
  const getModelName = (modelId) => {
    const modelMap = {
      'yolov8n': 'YOLOv8 Nano',
      'yolov8s': 'YOLOv8 Small',
      'yolov8m': 'YOLOv8 Medium',
      'yolov8l': 'YOLOv8 Large'
    };
    return modelMap[modelId] || modelId;
  };

  // Get camera status for specific pair
  const getCameraStatus = (pairId, cameraId) => {
    const stream = streamStatus[pairId];
    const stats = cameraStats[cameraId];
    
    if (stream?.error) return 'error';
    if (stream?.loading) return 'loading';
    if (stats?.cameraStatus === 'connected' && stream?.connected) return 'active';
    if (stream?.connected) return 'connected';
    return 'disconnected';
  };

  // Get alert severity for camera
  const getAlertSeverity = (cameraId) => {
    if (!alertedCameras.has(cameraId)) return null;
    
    const stats = cameraStats[cameraId];
    if (stats?.lastDetection) {
      const confidence = stats.lastDetection.confidence;
      const objectType = stats.lastDetection.class_name?.toLowerCase();
      
      // Determine severity based on object type and confidence
      if (objectType === 'person' && confidence >= 0.85) return 'critical';
      if ((objectType === 'person' && confidence >= 0.7) || 
          ['car', 'truck', 'motorcycle'].includes(objectType)) return 'warning';
    }
    
    return 'info';
  };

  // Format last detection time
  const formatLastDetection = (lastDetection) => {
    if (!lastDetection) return 'No recent detections';
    
    try {
      const time = new Date(lastDetection.timestamp).toLocaleTimeString();
      const confidence = Math.round(lastDetection.confidence * 100);
      return `${lastDetection.class_name} (${confidence}%) at ${time}`;
    } catch {
      return 'Invalid detection data';
    }
  };

  // Calculate grid layout styles
  const gridStyle = {
    display: 'grid',
    gridTemplateColumns: `repeat(${cols}, 1fr)`,
    gridTemplateRows: `repeat(${rows}, 1fr)`,
    gap: '16px',
    width: '100%',
    height: '100%'
  };

  return (
    <div className="enhanced-camera-grid-container">
      <div className="camera-grid" style={gridStyle}>
        {cameraModelPairs.map((pair, index) => {
          const cameraId = pair.camera.id;
          const pairId = pair.pairId || `${cameraId}_${pair.model}_${index}`;
          const isTracking = trackingPairs.some(tp => 
            tp.cameraId === cameraId && tp.model === pair.model
          );
          const status = getCameraStatus(pairId, cameraId);
          const isAlerted = alertedCameras.has(cameraId);
          const alertSeverity = getAlertSeverity(cameraId);
          const stats = cameraStats[cameraId];
          
          // Count instances of this camera
          const instanceNumber = cameraModelPairs
            .filter(p => p.camera.id === cameraId)
            .indexOf(pair) + 1;
          const totalInstances = cameraModelPairs
            .filter(p => p.camera.id === cameraId).length;
          
          return (
            <div 
              key={pairId} // ✅ Use unique pairId instead of cameraId
              className={`camera-cell ${status} ${isAlerted && flashEnabled ? 'alerted' : ''}`}
            >
              <div 
                className={`camera-frame ${
                  isAlerted && flashEnabled ? `${alertSeverity}-alert` : ''
                }`}
              >
                {/* Camera Feed */}
                <img 
                  ref={(el) => (imgRefs.current[pairId] = el)} // ✅ Use unique pairId for ref
                  alt={`Camera ${pair.camera.name || cameraId} - ${getModelName(pair.model)}`} 
                  className="camera-feed"
                />
                
                {/* Loading indicator */}
                {streamStatus[pairId]?.loading && (
                  <div className="stream-overlay loading">
                    <div className="loading-spinner"></div>
                    <span>Connecting...</span>
                  </div>
                )}
                
                {/* Error indicator */}
                {streamStatus[pairId]?.error && (
                  <div className="stream-overlay error">
                    <span className="error-icon">⚠️</span>
                    <span>Connection Lost</span>
                    <small>Attempting to reconnect...</small>
                  </div>
                )}
                
                {/* Camera Information Overlay */}
                <div className="camera-info-overlay">
                  <div className="camera-header">
                    <span className="camera-name">
                      {pair.camera.name || `Camera ${cameraId}`}
                      {totalInstances > 1 && (
                        <span className="instance-indicator">#{instanceNumber}</span>
                      )}
                    </span>
                    
                    <div className="status-indicators">
                      <div className={`connection-status ${status}`}>
                        <span className="status-dot"></span>
                      </div>
                      
                      {isTracking && (
                        <div className="tracking-indicator active">
                          <span className="tracking-icon">🎯</span>
                        </div>
                      )}
                      
                      {isAlerted && (
                        <div className={`alert-indicator ${alertSeverity}`}>
                          <span className="alert-icon">
                            {alertSeverity === 'critical' ? '🚨' : 
                             alertSeverity === 'warning' ? '⚠️' : '🔍'}
                          </span>
                        </div>
                      )}
                    </div>
                  </div>
                  
                  <div className="camera-footer">
                    <div className="model-info">
                      <span className={`model-badge ${isTracking ? 'active' : 'inactive'}`}>
                        {getModelName(pair.model)}
                      </span>
                    </div>
                    
                    {isTracking && stats && (
                      <div className="detection-stats">
                        <div className="stats-row">
                          <span className="stat-label">Detections:</span>
                          <span className="stat-value">{stats.totalDetections}</span>
                        </div>
                        
                        {stats.recentDetections > 0 && (
                          <div className="stats-row recent">
                            <span className="stat-label">Recent:</span>
                            <span className="stat-value">{stats.recentDetections}</span>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                </div>
                
                {/* Alert Banner */}
                {isAlerted && (
                  <div className={`alert-banner ${alertSeverity}`}>
                    <div className="alert-content">
                      <span className="alert-icon">
                        {alertSeverity === 'critical' ? '🚨' : 
                         alertSeverity === 'warning' ? '⚠️' : '🔍'}
                      </span>
                      <div className="alert-text">
                        <span className="alert-title">
                          {alertSeverity === 'critical' ? 'CRITICAL ALERT' :
                           alertSeverity === 'warning' ? 'WARNING' : 'DETECTION'}
                        </span>
                        {stats?.lastDetection && (
                          <span className="alert-details">
                            {stats.lastDetection.class_name} detected
                          </span>
                        )}
                      </div>
                    </div>
                  </div>
                )}
                
                {/* Last Detection Info */}
                {isTracking && stats?.lastDetection && !isAlerted && (
                  <div className="last-detection-info">
                    <small>{formatLastDetection(stats.lastDetection)}</small>
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>
      
      {/* Empty State */}
      {cameraModelPairs.length === 0 && (
        <div className="no-cameras-message">
          <div className="empty-state-icon">
            <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"></path>
              <circle cx="12" cy="13" r="4"></circle>
            </svg>
          </div>
          <h3>No Cameras Configured</h3>
          <p>Add camera-model pairs from the control panel to start monitoring.</p>
          <div className="setup-steps">
            <div className="step">
              <span className="step-number">1</span>
              <span>Select a camera from the dropdown</span>
            </div>
            <div className="step">
              <span className="step-number">2</span>
              <span>Choose an AI detection model</span>
            </div>
            <div className="step">
              <span className="step-number">3</span>
              <span>Click "Add" to create the pair</span>
            </div>
            <div className="step">
              <span className="step-number">4</span>
              <span>Start tracking to begin monitoring</span>
            </div>
          </div>
        </div>
      )}
      
      {/* Enhanced Grid Stats Footer */}
      {cameraModelPairs.length > 0 && (
        <div className="grid-stats-footer">
          <div className="stats-section">
            <span className="stats-label">Configurations:</span>
            <span className="stats-value">{cameraModelPairs.length}</span>
          </div>
          
          <div className="stats-section">
            <span className="stats-label">Unique Cameras:</span>
            <span className="stats-value">
              {new Set(cameraModelPairs.map(p => p.camera.id)).size}
            </span>
          </div>
          
          {isTrackingActive && (
            <>
              <div className="stats-section">
                <span className="stats-label">Tracking:</span>
                <span className="stats-value">{trackingPairs.length}</span>
              </div>
              
              <div className="stats-section">
                <span className="stats-label">Active Alerts:</span>
                <span className="stats-value">{alertedCameras.size}</span>
              </div>
              
              <div className="stats-section">
                <span className="stats-label">Total Detections:</span>
                <span className="stats-value">
                  {Object.values(cameraStats).reduce((sum, stats) => 
                    sum + (stats?.totalDetections || 0), 0
                  )}
                </span>
              </div>
            </>
          )}
          
          <div className="stats-section">
            <span className="stats-label">Layout:</span>
            <span className="stats-value">{rows}×{cols}</span>
          </div>
        </div>
      )}
    </div>
  );
}

export default CameraGrid;