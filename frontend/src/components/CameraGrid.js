// Enhanced CameraGrid.js - Electron Fullview Integration
import React, { useEffect, useRef, useState, useCallback } from 'react';
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
  const [selectedImg, setSelectedImg] = useState(null);
  const [isFullviewOpen, setIsFullviewOpen] = useState(false);
  const [activeFullviewWindows, setActiveFullviewWindows] = useState([]);
  const fullviewImgRef = useRef(null);
  const { rows, cols } = gridLayout;

  // Enhanced fullview handler with Electron support
  const handleFullView = useCallback(async (pairId) => {
    const pair = cameraModelPairs.find(p => 
      (p.pairId || `${p.camera.id}_${p.model}_${cameraModelPairs.indexOf(p)}`) === pairId
    );
    
    if (!pair) return;

    // Check if Electron fullview is available
    if (window.electron?.openFullview) {
      try {
        const isTracking = trackingPairs.some(tp => 
          tp.cameraId === pair.camera.id && tp.model === pair.model
        );
        
        const timestamp = Date.now();
        const streamUrl = isTracking
          ? `http://localhost:8001/api/cameras/${pair.camera.id}/detection_stream?model=${pair.model}&pair_id=${pairId}&session=fullview_${pairId}&t=${timestamp}`
          : `http://localhost:8001/api/cameras/${pair.camera.id}/stream?pair_id=${pairId}&session=fullview_${pairId}&t=${timestamp}`;

        console.log(`🪟 Opening Electron fullview window for camera ${pair.camera.id}`);
        
        const result = await window.electron.openFullview({
          camera: pair.camera,
          model: pair.model,
          streamUrl: streamUrl,
          pairId: pairId,
          isTracking: isTracking
        });
        
        if (result.success) {
          console.log('✅ Fullview window opened:', result.windowId);
          
          // Update active windows list
          setActiveFullviewWindows(prev => [...prev, {
            id: result.windowId,
            cameraId: pair.camera.id,
            pairId: pairId,
            cameraName: pair.camera.name,
            model: pair.model
          }]);
          
          // Play success sound if available
          if (window.electron.playBeep) {
            window.electron.playBeep();
          }
        } else {
          console.warn('⚠️ Failed to open Electron fullview, falling back to modal');
          openModalFullview(pairId, pair);
        }
      } catch (error) {
        console.error('❌ Electron fullview error:', error);
        openModalFullview(pairId, pair);
      }
    } else {
      // Fallback to modal fullview
      console.log('📱 Using modal fullview (Electron not available)');
      openModalFullview(pairId, pair);
    }
  }, [cameraModelPairs, trackingPairs]);

  // Fallback modal fullview
  const openModalFullview = useCallback((pairId, pair) => {
    setSelectedImg({ pairId, pair });
    setIsFullviewOpen(true);
  }, []);

  // Close modal fullview
  const closeFullView = useCallback(() => {
    setIsFullviewOpen(false);
    setSelectedImg(null);
    if (fullviewImgRef.current) {
      fullviewImgRef.current.src = '';
      fullviewImgRef.current.onload = null;
      fullviewImgRef.current.onerror = null;
    }
  }, []);

  // Listen for Electron fullview window events
  useEffect(() => {
    if (window.electron?.on) {
      const handleWindowClosed = (windowId) => {
        console.log('🔒 Fullview window closed:', windowId);
        setActiveFullviewWindows(prev => prev.filter(w => w.id !== windowId));
      };

      const handleWindowCreated = (windowData) => {
        console.log('🆕 Fullview window created:', windowData);
        // Window data will be added when handleFullView succeeds
      };

      window.electron.on('fullview-window-closed', handleWindowClosed);
      window.electron.on('fullview-window-created', handleWindowCreated);

      // Cleanup listeners
      return () => {
        if (window.electron.removeListener) {
          window.electron.removeListener('fullview-window-closed', handleWindowClosed);
          window.electron.removeListener('fullview-window-created', handleWindowCreated);
        }
      };
    }
  }, []);

  // Load existing fullview windows on mount
  useEffect(() => {
    const loadActiveWindows = async () => {
      if (window.electron?.getFullviewWindows) {
        try {
          const windows = await window.electron.getFullviewWindows();
          setActiveFullviewWindows(windows);
          console.log('📋 Loaded active fullview windows:', windows.length);
        } catch (error) {
          console.warn('Failed to load active windows:', error);
        }
      }
    };

    loadActiveWindows();
  }, []);

  // Enhanced keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e) => {
      // Only handle if not typing in an input
      if (e.target.tagName.toLowerCase() === 'input') return;
      
      // ESC to close modal
      if (e.key === 'Escape' && isFullviewOpen) {
        closeFullView();
        return;
      }
      
      // Ctrl/Cmd + number keys to open specific camera fullview
      if ((e.ctrlKey || e.metaKey) && !isNaN(parseInt(e.key)) && e.key !== '0') {
        const index = parseInt(e.key) - 1;
        if (index < cameraModelPairs.length) {
          e.preventDefault();
          const pair = cameraModelPairs[index];
          const pairId = pair.pairId || `${pair.camera.id}_${pair.model}_${index}`;
          handleFullView(pairId);
        }
      }
      
      // Ctrl/Cmd + A to open all cameras (if Electron available)
      if ((e.ctrlKey || e.metaKey) && e.key === 'a' && window.electron?.openFullview) {
        e.preventDefault();
        openAllCameras();
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [isFullviewOpen, cameraModelPairs, handleFullView, closeFullView]);

  // Open all cameras in separate windows
  const openAllCameras = useCallback(async () => {
    if (!window.electron?.openFullview) {
      console.warn('Electron fullview not available for batch operation');
      return;
    }

    console.log('🎬 Opening all cameras in separate windows...');
    
    for (const [index, pair] of cameraModelPairs.entries()) {
      const pairId = pair.pairId || `${pair.camera.id}_${pair.model}_${index}`;
      await handleFullView(pairId);
      // Small delay to prevent overwhelming the system
      await new Promise(resolve => setTimeout(resolve, 200));
    }
  }, [cameraModelPairs, handleFullView]);

  // Close all fullview windows
  const closeAllWindows = useCallback(async () => {
    if (!window.electron?.closeFullview) return;

    console.log('🔒 Closing all fullview windows...');
    
    for (const window of activeFullviewWindows) {
      try {
        await window.electron.closeFullview(window.id);
      } catch (error) {
        console.warn('Failed to close window:', window.id, error);
      }
    }
  }, [activeFullviewWindows]);

  // Check if camera has active fullview window
  const hasActiveFullview = useCallback((cameraId, model) => {
    return activeFullviewWindows.some(w => 
      w.cameraId === cameraId && w.model === model
    );
  }, [activeFullviewWindows]);

  // Setup fullview stream when modal opens
  useEffect(() => {
    if (isFullviewOpen && selectedImg && fullviewImgRef.current) {
      const { pairId, pair } = selectedImg;
      const camera = pair.camera;
      const imgElement = fullviewImgRef.current;
      
      const isTracking = trackingPairs.some(tp => 
        tp.cameraId === camera.id && tp.model === pair.model
      );
      
      const model = isTracking 
        ? trackingPairs.find(tp => tp.cameraId === camera.id && tp.model === pair.model)?.model 
        : pair.model;
      
      const timestamp = Date.now();
      const streamUrl = isTracking
        ? `http://localhost:8001/api/cameras/${camera.id}/detection_stream?model=${model}&pair_id=${pairId}&session=modal_${pairId}&t=${timestamp}`
        : `http://localhost:8001/api/cameras/${camera.id}/stream?pair_id=${pairId}&session=modal_${pairId}&t=${timestamp}`;
      
      console.log(`Setting modal fullview stream URL for camera ${camera.id} (${model}):`, streamUrl);
      
      imgElement.onload = () => {
        console.log(`Modal fullview stream loaded for camera ${camera.id}`);
      };
      
      imgElement.onerror = (error) => {
        console.warn(`Modal fullview stream error for camera ${camera.id}:`, error);
      };
      
      imgElement.src = streamUrl;
    }
  }, [isFullviewOpen, selectedImg, trackingPairs]);

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
        const isTracking = trackingPairs.some(tp => 
          tp.cameraId === camera.id && tp.model === pair.model
        );
        
        const model = isTracking 
          ? trackingPairs.find(tp => tp.cameraId === camera.id && tp.model === pair.model)?.model 
          : pair.model;
        
        setStreamStatus(prev => ({
          ...prev,
          [pairId]: { loading: true, error: false, lastUpdate: Date.now() }
        }));
        
        const timestamp = Date.now();
        const streamUrl = isTracking
          ? `http://localhost:8001/api/cameras/${camera.id}/detection_stream?model=${model}&pair_id=${pairId}&session=${pairId}&t=${timestamp}`
          : `http://localhost:8001/api/cameras/${camera.id}/stream?pair_id=${pairId}&session=${pairId}&t=${timestamp}`;
        
        console.log(`Setting stream URL for camera ${camera.id} (${model}):`, streamUrl);
        imgElement.src = streamUrl;
        
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
          
          if (streamErrorTimers.current[pairId]) {
            clearTimeout(streamErrorTimers.current[pairId]);
          }
          
          const retryDelay = Math.min(2000 * Math.pow(2, (prev?.[pairId]?.retryCount || 0)), 30000);
          
          streamErrorTimers.current[pairId] = setTimeout(() => {
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

    return () => {
      cameraModelPairs.forEach((pair, index) => {
        const pairId = pair.pairId || `${pair.camera.id}_${pair.model}_${index}`;
        const imgElement = imgRefs.current[pairId];
        if (imgElement) {
          imgElement.onload = null;
          imgElement.onerror = null;
          imgElement.src = '';
        }
        
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

    fetchStats();
    const statsInterval = setInterval(fetchStats, 5000);
    
    return () => clearInterval(statsInterval);
  }, [isTrackingActive, cameraModelPairs]);

  // Utility functions
  const getModelName = (modelId) => {
    const modelMap = {
      'yolov8n': 'YOLOv8 Nano',
      'yolov8s': 'YOLOv8 Small',
      'yolov8m': 'YOLOv8 Medium',
      'yolov8l': 'YOLOv8 Large'
    };
    return modelMap[modelId] || modelId;
  };

  const getCameraStatus = (pairId, cameraId) => {
    const stream = streamStatus[pairId];
    const stats = cameraStats[cameraId];
    
    if (stream?.error) return 'error';
    if (stream?.loading) return 'loading';
    if (stats?.cameraStatus === 'connected' && stream?.connected) return 'active';
    if (stream?.connected) return 'connected';
    return 'disconnected';
  };

  const getAlertSeverity = (cameraId) => {
    if (!alertedCameras.has(cameraId)) return null;
    
    const stats = cameraStats[cameraId];
    if (stats?.lastDetection) {
      const confidence = stats.lastDetection.confidence;
      const objectType = stats.lastDetection.class_name?.toLowerCase();
      
      if (objectType === 'person' && confidence >= 0.85) return 'critical';
      if ((objectType === 'person' && confidence >= 0.7) || 
          ['car', 'truck', 'motorcycle'].includes(objectType)) return 'warning';
    }
    
    return 'info';
  };

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
      {/* Enhanced Fullview Controls Bar */}
      {(activeFullviewWindows.length > 0 || window.electron?.openFullview) && (
        <div className="fullview-controls-bar" style={{
          position: 'absolute',
          top: '10px',
          right: '10px',
          zIndex: 1000,
          display: 'flex',
          gap: '8px',
          padding: '8px',
          background: 'rgba(0, 0, 0, 0.7)',
          borderRadius: '6px',
          fontSize: '12px'
        }}>
          {window.electron?.openFullview && (
            <button
              onClick={openAllCameras}
              disabled={cameraModelPairs.length === 0}
              style={{
                background: 'rgba(33, 150, 243, 0.8)',
                border: 'none',
                color: 'white',
                padding: '6px 10px',
                borderRadius: '4px',
                cursor: 'pointer',
                fontSize: '11px'
              }}
              title="Open all cameras in separate windows (Ctrl+A)"
            >
              🪟 Open All ({cameraModelPairs.length})
            </button>
          )}
          
          {activeFullviewWindows.length > 0 && (
            <>
              <span style={{ color: 'white', alignSelf: 'center' }}>
                {activeFullviewWindows.length} window{activeFullviewWindows.length > 1 ? 's' : ''} open
              </span>
              <button
                onClick={closeAllWindows}
                style={{
                  background: 'rgba(244, 67, 54, 0.8)',
                  border: 'none',
                  color: 'white',
                  padding: '6px 10px',
                  borderRadius: '4px',
                  cursor: 'pointer',
                  fontSize: '11px'
                }}
                title="Close all fullview windows"
              >
                ✕ Close All
              </button>
            </>
          )}
        </div>
      )}

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
          const hasFullview = hasActiveFullview(cameraId, pair.model);
          
          const instanceNumber = cameraModelPairs
            .filter(p => p.camera.id === cameraId)
            .indexOf(pair) + 1;
          const totalInstances = cameraModelPairs
            .filter(p => p.camera.id === cameraId).length;
          
          return (
            <div 
              key={pairId}
              className={`camera-cell ${status} ${isAlerted && flashEnabled ? 'alerted' : ''} ${hasFullview ? 'has-fullview' : ''}`}
            >
              <div 
                className={`camera-frame ${
                  isAlerted && flashEnabled ? `${alertSeverity}-alert` : ''
                }`}
              >
                <img 
                  ref={(el) => (imgRefs.current[pairId] = el)}
                  alt={`Camera ${pair.camera.name || cameraId} - ${getModelName(pair.model)}`} 
                  className="camera-feed"
                  onClick={() => handleFullView(pairId)}
                  style={{ cursor: 'pointer' }}
                />
             
                {/* Enhanced Fullview Button Overlay */}
                <div className="fullview-overlay" onClick={() => handleFullView(pairId)}>
                  <div className="fullview-button">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <path d="M8 3H5a2 2 0 0 0-2 2v3m18 0V5a2 2 0 0 0-2-2h-3m0 18h3a2 2 0 0 0 2-2v-3M3 16v3a2 2 0 0 0 2 2h3"/>
                    </svg>
                    <span>
                      {window.electron?.openFullview ? 'Open Window' : 'Full View'}
                      {hasFullview && ' (Open)'}
                    </span>
                  </div>
                </div>

                {/* Fullview Status Indicator */}
                {hasFullview && (
                  <div className="fullview-status-indicator" style={{
                    position: 'absolute',
                    top: '8px',
                    right: '8px',
                    background: 'rgba(76, 175, 80, 0.9)',
                    color: 'white',
                    padding: '4px 8px',
                    borderRadius: '4px',
                    fontSize: '10px',
                    fontWeight: 'bold'
                  }}>
                    🪟 WINDOW OPEN
                  </div>
                )}

                {/* Keyboard Shortcut Hint */}
                {index < 9 && (
                  <div className="keyboard-hint" style={{
                    position: 'absolute',
                    top: '8px',
                    left: '8px',
                    background: 'rgba(0, 0, 0, 0.7)',
                    color: 'white',
                    padding: '2px 6px',
                    borderRadius: '3px',
                    fontSize: '10px',
                    opacity: 0.7
                  }}>
                    Ctrl+{index + 1}
                  </div>
                )}

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
                    <span className="camera-name-grid">
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

      {/* Enhanced Fullview Modal (Fallback) */}
      {isFullviewOpen && selectedImg && (
        <div className="fullview-modal" onClick={closeFullView}>
          <div className="fullview-content" onClick={(e) => e.stopPropagation()}>
            <div className="fullview-header">
              <div className="fullview-title">
                <h2>
                  {selectedImg.pair.camera.name || `Camera ${selectedImg.pair.camera.id}`}
                  <span className="fullview-model">{getModelName(selectedImg.pair.model)}</span>
                </h2>
                {!window.electron?.openFullview && (
                  <small style={{ color: '#888', fontSize: '12px' }}>
                    Modal Mode (Electron window not available)
                  </small>
                )}
              </div>
              
              <div className="fullview-controls">
                {window.electron?.openFullview && (
                  <button 
                    className="fullview-detach-btn"
                    onClick={() => {
                      closeFullView();
                      setTimeout(() => handleFullView(selectedImg.pairId), 100);
                    }}
                    title="Open in separate window"
                    style={{
                      background: 'rgba(33, 150, 243, 0.2)',
                      border: '1px solid rgba(33, 150, 243, 0.5)',
                      color: 'white',
                      padding: '6px 12px',
                      borderRadius: '4px',
                      cursor: 'pointer',
                      fontSize: '12px',
                      marginRight: '8px'
                    }}
                  >
                    🪟 Detach
                  </button>
                )}
                
                <button 
                  className="fullview-close-btn"
                  onClick={closeFullView}
                  title="Close (ESC)"
                >
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <line x1="18" y1="6" x2="6" y2="18"></line>
                    <line x1="6" y1="6" x2="18" y2="18"></line>
                  </svg>
                </button>
              </div>
            </div>
            
            <div className="fullview-body">
              <div className="fullview-video-container">
                <img 
                  ref={fullviewImgRef}
                  alt={`Full view of ${selectedImg.pair.camera.name || selectedImg.pair.camera.id}`}
                  className="fullview-video"
                />
                
                <div className="fullview-info-overlay">
                  <div className="fullview-status">
                    <div className={`connection-status ${getCameraStatus(selectedImg.pairId, selectedImg.pair.camera.id)}`}>
                      <span className="status-dot"></span>
                      <span className="status-text">
                        {getCameraStatus(selectedImg.pairId, selectedImg.pair.camera.id).toUpperCase()}
                      </span>
                    </div>
                    
                    {trackingPairs.some(tp => tp.cameraId === selectedImg.pair.camera.id && tp.model === selectedImg.pair.model) && (
                      <div className="tracking-status active">
                        <span className="tracking-icon">🎯</span>
                        <span>AI TRACKING ACTIVE</span>
                      </div>
                    )}
                    
                    {/* Modal-specific status */}
                    <div className="modal-status" style={{
                      background: 'rgba(156, 39, 176, 0.8)',
                      color: 'white',
                      padding: '4px 8px',
                      borderRadius: '4px',
                      fontSize: '11px'
                    }}>
                      📱 MODAL VIEW
                    </div>
                  </div>
                </div>

                {/* Enhanced Modal Controls */}
                <div className="modal-controls" style={{
                  position: 'absolute',
                  bottom: '20px',
                  left: '50%',
                  transform: 'translateX(-50%)',
                  display: 'flex',
                  gap: '8px',
                  background: 'rgba(0, 0, 0, 0.7)',
                  padding: '8px 12px',
                  borderRadius: '6px'
                }}>
                  {window.electron?.takeScreenshot && (
                    <button
                      onClick={() => window.electron.takeScreenshot()}
                      style={{
                        background: 'rgba(76, 175, 80, 0.2)',
                        border: '1px solid rgba(76, 175, 80, 0.5)',
                        color: 'white',
                        padding: '6px 10px',
                        borderRadius: '4px',
                        cursor: 'pointer',
                        fontSize: '11px'
                      }}
                      title="Take Screenshot"
                    >
                      📸 Screenshot
                    </button>
                  )}
                  
                  <button
                    onClick={() => {
                      // Take manual screenshot as fallback
                      const canvas = document.createElement('canvas');
                      const video = fullviewImgRef.current;
                      if (video) {
                        canvas.width = video.naturalWidth || 1920;
                        canvas.height = video.naturalHeight || 1080;
                        const ctx = canvas.getContext('2d');
                        ctx.drawImage(video, 0, 0);
                        
                        const link = document.createElement('a');
                        link.download = `camera_${selectedImg.pair.camera.id}_${Date.now()}.png`;
                        link.href = canvas.toDataURL();
                        link.click();
                      }
                    }}
                    style={{
                      background: 'rgba(255, 152, 0, 0.2)',
                      border: '1px solid rgba(255, 152, 0, 0.5)',
                      color: 'white',
                      padding: '6px 10px',
                      borderRadius: '4px',
                      cursor: 'pointer',
                      fontSize: '11px'
                    }}
                    title="Save Image"
                  >
                    💾 Save
                  </button>
                  
                  <span style={{
                    color: 'white',
                    fontSize: '10px',
                    alignSelf: 'center',
                    padding: '0 8px',
                    borderLeft: '1px solid rgba(255, 255, 255, 0.3)'
                  }}>
                    ESC to close
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

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
          
          {/* Enhanced setup info for Electron */}
          {window.electron?.openFullview && (
            <div className="electron-features" style={{
              marginTop: '20px',
              padding: '16px',
              background: 'rgba(33, 150, 243, 0.1)',
              borderRadius: '8px',
              border: '1px solid rgba(33, 150, 243, 0.3)'
            }}>
              <h4 style={{ color: '#2196F3', margin: '0 0 8px 0' }}>🪟 Enhanced Desktop Features</h4>
              <ul style={{ fontSize: '14px', color: '#666', margin: 0, paddingLeft: '20px' }}>
                <li>Open cameras in separate windows</li>
                <li>Multi-monitor support</li>
                <li>Keyboard shortcuts (Ctrl+1-9)</li>
                <li>Native screenshots and recording</li>
                <li>Always-on-top windows</li>
              </ul>
            </div>
          )}
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
          
          {/* Fullview windows stats */}
          {activeFullviewWindows.length > 0 && (
            <div className="stats-section">
              <span className="stats-label">Fullview Windows:</span>
              <span className="stats-value">{activeFullviewWindows.length}</span>
            </div>
          )}
          
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

          {/* Desktop mode indicator */}
          {window.electron?.openFullview && (
            <div className="stats-section">
              <span className="stats-label">Mode:</span>
              <span className="stats-value" style={{ color: '#2196F3' }}>🪟 Desktop</span>
            </div>
          )}
        </div>
      )}

      {/* Help overlay for keyboard shortcuts */}
      {cameraModelPairs.length > 0 && window.electron?.openFullview && (
        <div className="keyboard-help" style={{
          position: 'absolute',
          bottom: '10px',
          left: '10px',
          background: 'rgba(0, 0, 0, 0.7)',
          color: 'white',
          padding: '8px 12px',
          borderRadius: '6px',
          fontSize: '11px',
          opacity: 0.8,
          zIndex: 1000
        }}>
          <div style={{ fontWeight: 'bold', marginBottom: '4px' }}>🔧 Shortcuts:</div>
          <div>Ctrl+1-9: Open camera • Ctrl+A: Open all • ESC: Close modal</div>
        </div>
      )}
    </div>
  );
}

export default CameraGrid;