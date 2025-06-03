// Enhanced ControlPanel.js - Same UI, Enhanced Functionality
import React, { useState } from 'react';
import './ControlPanel.css';

const ControlPanel = ({
  cameras,
  cameraModelPairs,
  addCameraModelPair,
  removeCameraModelPair,
  isTracking,
  startTracking,
  stopTracking,
  outputDir,
  selectOutputDirectory,
  updateGridLayout,
  onToggleAlerts,
  onToggleHistory,
  onToggleDashboard,
  onToggleDetectionStatus,
  alertCount = 0,
  criticalAlertCount = 0,
  activeAlertConnections,
  showDetectionStatus,
  soundEnabled,
  setSoundEnabled,
  flashEnabled,
  setFlashEnabled
}) => {
  const [isExpanded, setIsExpanded] = useState(true);
  const [rows, setRows] = useState(2);
  const [cols, setCols] = useState(2);
  const [selectedCamera, setSelectedCamera] = useState('');
  const [selectedModel, setSelectedModel] = useState('yolov8n');
  const [activeTab, setActiveTab] = useState('cameras');

  // Available YOLO models
  const models = [
    { id: 'yolov8n', name: 'YOLOv8 Nano' },
    { id: 'yolov8s', name: 'YOLOv8 Small' },
    { id: 'yolov8m', name: 'YOLOv8 Medium' },
    { id: 'yolov8l', name: 'YOLOv8 Large' },
  ];

  // Toggle panel expansion
  const togglePanel = () => {
    setIsExpanded(!isExpanded);
  };

  // Update grid layout
  const handleLayoutChange = () => {
    updateGridLayout(parseInt(rows), parseInt(cols));
  };

  // Add camera-model pair (Enhanced: Allow same camera with different models)
  const handleAddPair = () => {
    if (selectedCamera) {
      const cameraToAdd = cameras.find(cam => cam.id === selectedCamera);
      if (cameraToAdd) {
        // Check if this exact camera-model combination already exists
        const existingPair = cameraModelPairs.find(pair => 
          pair.camera.id === cameraToAdd.id && pair.model === selectedModel
        );
        
        if (!existingPair) {
          addCameraModelPair({
            camera: cameraToAdd,
            model: selectedModel,
            pairId: `${cameraToAdd.id}_${selectedModel}_${Date.now()}` // Unique identifier
          });
          setSelectedCamera('');
        } else {
          // Show warning if exact combination already exists
          alert(`This camera is already configured with ${models.find(m => m.id === selectedModel)?.name || selectedModel}`);
        }
      }
    }
  };

  // Get all cameras (no filtering - allow multiple configurations)
  const getAvailableCameras = () => {
    return cameras; // Return all cameras, no filtering
  };

  // Get system status
  const getSystemStatus = () => {
    if (!isTracking) return { status: 'offline', color: '#6c757d' };
    
    const activeConnections = Object.keys(activeAlertConnections).length;
    const totalCameras = cameraModelPairs.length;
    
    if (activeConnections === totalCameras && totalCameras > 0) {
      return { status: 'optimal', color: '#28a745' };
    } else if (activeConnections > 0) {
      return { status: 'partial', color: '#ffc107' };
    } else {
      return { status: 'connecting', color: '#007bff' };
    }
  };

  // Enhanced: Update model for existing pair (Modified to handle multiple instances)
  const updatePairModel = (pairId, newModel) => {
    if (isTracking) return; // Don't allow changes during tracking
    
    const existingPair = cameraModelPairs.find(pair => 
      (pair.pairId || `${pair.camera.id}_${pair.model}`) === pairId
    );
    
    if (existingPair) {
      // Check if this camera-model combination would be duplicate
      const wouldBeDuplicate = cameraModelPairs.some(pair => 
        pair.camera.id === existingPair.camera.id && 
        pair.model === newModel && 
        (pair.pairId || `${pair.camera.id}_${pair.model}`) !== pairId
      );
      
      if (wouldBeDuplicate) {
        alert(`This camera already has a configuration with ${models.find(m => m.id === newModel)?.name || newModel}`);
        return;
      }
      
      // Remove and re-add with new model
      removeCameraModelPair(pairId);
      setTimeout(() => {
        addCameraModelPair({
          camera: existingPair.camera,
          model: newModel,
          pairId: `${existingPair.camera.id}_${newModel}_${Date.now()}`
        });
      }, 0);
    }
  };

  // Enhanced: Get model statistics (Updated for multiple instances)
  const getModelStatistics = () => {
    const modelCounts = {};
    const cameraModelCombinations = {};
    
    cameraModelPairs.forEach(pair => {
      modelCounts[pair.model] = (modelCounts[pair.model] || 0) + 1;
      
      const cameraName = pair.camera.name || `Camera ${pair.camera.id}`;
      if (!cameraModelCombinations[cameraName]) {
        cameraModelCombinations[cameraName] = [];
      }
      cameraModelCombinations[cameraName].push(pair.model);
    });
    
    return { modelCounts, cameraModelCombinations };
  };

  // Enhanced: Quick add all available cameras with same model (NEW FEATURE)
  const handleQuickAddAll = () => {
    if (isTracking) return;
    
    const availableCameras = getAvailableCameras();
    availableCameras.forEach(camera => {
      addCameraModelPair({
        camera,
        model: selectedModel
      });
    });
  };

  const systemStatus = getSystemStatus();
  const availableCameras = getAvailableCameras();
  const { modelCounts, cameraModelCombinations } = getModelStatistics();

  return (
    <div className={`control-panel ${isExpanded ? 'expanded' : 'collapsed'}`}>
      <button className="toggle-button" onClick={togglePanel}>
        {isExpanded ? '◀' : '▶'}
      </button>

      {isExpanded && (
        <div className="panel-content">
          {/* System Status Header */}
          <div className="system-status-header">
            <div className="status-indicator-group">
              <div 
                className={`system-status-dot ${systemStatus.status}`}
                style={{ backgroundColor: systemStatus.color }}
              />
              <span className="system-status-text">
                System: {systemStatus.status.charAt(0).toUpperCase() + systemStatus.status.slice(1)}
              </span>
            </div>
            
            {isTracking && (
              <div className="connection-stats">
                <span>{Object.keys(activeAlertConnections).length}/{cameraModelPairs.length} Active</span>
              </div>
            )}
          </div>

          {/* Critical Alert Banner */}
          {criticalAlertCount > 0 && (
            <div className="critical-alert-banner">
              <span className="alert-icon">🚨</span>
              <span className="alert-text">
                {criticalAlertCount} Critical Alert{criticalAlertCount > 1 ? 's' : ''}
              </span>
              <button 
                className="view-alerts-btn"
                onClick={onToggleAlerts}
              >
                View
              </button>
            </div>
          )}

          {/* Tab Navigation */}
          <div className="tab-navigation">
            <button 
              className={`tab-button ${activeTab === 'cameras' ? 'active' : ''}`}
              onClick={() => setActiveTab('cameras')}
            >
              📹 Cameras
            </button>
            <button 
              className={`tab-button ${activeTab === 'tracking' ? 'active' : ''}`}
              onClick={() => setActiveTab('tracking')}
            >
              🎯 Tracking
            </button>
            <button 
              className={`tab-button ${activeTab === 'alerts' ? 'active' : ''}`}
              onClick={() => setActiveTab('alerts')}
            >
              🔔 Alerts
              {alertCount > 0 && (
                <span className="tab-badge">{alertCount}</span>
              )}
            </button>
            <button 
              className={`tab-button ${activeTab === 'settings' ? 'active' : ''}`}
              onClick={() => setActiveTab('settings')}
            >
              ⚙️ Settings
            </button>
          </div>

          {/* Tab Content */}
          <div className="tab-content">
            {/* Cameras Tab - Enhanced but same UI */}
            {activeTab === 'cameras' && (
              <div className="tab-panel">
                <h3>Camera Configuration</h3>
                
                <div className="camera-setup-section">
                  <div className="setup-form">
                    <div className="form-row">
                      <div className="form-group">
                        <label>Camera:</label>
                        <select 
                          value={selectedCamera} 
                          onChange={(e) => setSelectedCamera(e.target.value)}
                          disabled={isTracking}
                          className="form-select"
                        >
                          <option value="">Select a camera</option>
                          {availableCameras.map(camera => (
                            <option key={camera.id} value={camera.id}>
                              {camera.name || `Camera ${camera.id}`}
                            </option>
                          ))}
                        </select>
                      </div>
                      
                      <div className="form-group">
                        <label>AI Model:</label>
                        <select 
                          value={selectedModel} 
                          onChange={(e) => setSelectedModel(e.target.value)} 
                          disabled={isTracking}
                          className="form-select"
                        >
                          {models.map((model) => (
                            <option key={model.id} value={model.id}>
                              {model.name}
                            </option>
                          ))}
                        </select>
                      </div>
                    </div>
                    
                    <div className="form-row">
                      <button 
                        onClick={handleAddPair} 
                        disabled={!selectedCamera || isTracking}
                        className="add-pair-button"
                      >
                        <span className="button-icon">➕</span>
                        Add Camera Pair
                      </button>
                    </div>
                  </div>
                </div>
                
                <div className="selected-pairs-section">
                  <h4>Selected Camera-Model Pairs ({cameraModelPairs.length})</h4>
                  
                  {/* Enhanced: Model distribution info (Updated for multiple instances) */}
                  {cameraModelPairs.length > 0 && (
                    <div className="model-distribution" style={{ marginBottom: '16px' }}>
                      <small style={{ color: '#6c757d' }}>
                        Total configurations: {cameraModelPairs.length} | 
                        Models in use: {Object.entries(modelCounts).map(([model, count]) => 
                          `${models.find(m => m.id === model)?.name || model} (${count})`
                        ).join(', ')}
                      </small>
                    </div>
                  )}
                  
                  {cameraModelPairs.length === 0 ? (
                    <div className="no-pairs-message">
                      <div className="info-card">
                        <span className="info-icon">ℹ️</span>
                        <p>No camera-model pairs selected. Add at least one pair to start tracking.</p>
                      </div>
                    </div>
                  ) : (
                    <div className="pairs-list">
                      {cameraModelPairs.map((pair, index) => {
                        const pairId = pair.pairId || `${pair.camera.id}_${pair.model}_${index}`;
                        const isConnected = activeAlertConnections[pair.camera.id];
                        
                        return (
                          <div key={pairId} className="pair-card">
                            <div className="pair-info">
                              <div className="camera-details">
                                <span className="camera-name">
                                  {pair.camera.name || `Camera ${pair.camera.id}`}
                                </span>
                                
                                {/* Enhanced: Inline model selector (Updated for multiple instances) */}
                                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginTop: '4px' }}>
                                  <span style={{ fontSize: '12px', color: '#6c757d' }}>Model:</span>
                                  <select
                                    value={pair.model}
                                    onChange={(e) => updatePairModel(pairId, e.target.value)}
                                    disabled={isTracking}
                                    style={{
                                      fontSize: '11px',
                                      padding: '2px 4px',
                                      border: '1px solid #dee2e6',
                                      borderRadius: '3px',
                                      background: isTracking ? '#e9ecef' : 'white'
                                    }}
                                  >
                                    {models.map(model => (
                                      <option key={model.id} value={model.id}>
                                        {model.name}
                                      </option>
                                    ))}
                                  </select>
                                </div>
                                
                                {/* Show instance number if camera has multiple configurations */}
                                {cameraModelPairs.filter(p => p.camera.id === pair.camera.id).length > 1 && (
                                  <div style={{ fontSize: '10px', color: '#6c757d', marginTop: '2px' }}>
                                    Instance #{cameraModelPairs.filter(p => p.camera.id === pair.camera.id).indexOf(pair) + 1}
                                  </div>
                                )}
                              </div>
                              {isTracking && (
                                <div className={`connection-status ${isConnected ? 'connected' : 'disconnected'}`}>
                                  <span className="status-dot"></span>
                                  <span className="status-text">
                                    {isConnected ? 'Active' : 'Connecting...'}
                                  </span>
                                </div>
                              )}
                            </div>
                            <button 
                              onClick={() => removeCameraModelPair(pairId)} 
                              disabled={isTracking}
                              className="remove-button"
                              title="Remove this camera-model configuration"
                            >
                              ✕
                            </button>
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>

                <div className="layout-section">
                  <h4>Display Layout</h4>
                  <div className="layout-controls">
                    <div className="layout-inputs">
                      <div className="input-group">
                        <label>Rows:</label>
                        <input
                          type="number"
                          min="1"
                          max="4"
                          value={rows}
                          onChange={(e) => setRows(e.target.value)}
                          disabled={isTracking}
                          className="layout-input"
                        />
                      </div>
                      <div className="input-group">
                        <label>Columns:</label>
                        <input
                          type="number"
                          min="1"
                          max="4"
                          value={cols}
                          onChange={(e) => setCols(e.target.value)}
                          disabled={isTracking}
                          className="layout-input"
                        />
                      </div>
                    </div>
                    <button 
                      onClick={handleLayoutChange} 
                      disabled={isTracking}
                      className="update-layout-button"
                    >
                      Update Layout
                    </button>
                  </div>
                </div>
              </div>
            )}

            {/* Tracking Tab */}
            {activeTab === 'tracking' && (
              <div className="tab-panel">
                <h3>Tracking Controls</h3>
                
                <div className="tracking-status-card">
                  <div className="status-info">
                    <div className="status-row">
                      <span className="status-label">System Status:</span>
                      <span className={`status-value ${systemStatus.status}`}>
                        {systemStatus.status.charAt(0).toUpperCase() + systemStatus.status.slice(1)}
                      </span>
                    </div>
                    
                    {isTracking && (
                      <div className="tracking-stats">
                        <div className="stat-item">
                          <span className="stat-label">Active Cameras:</span>
                          <span className="stat-value">
                            {Object.keys(activeAlertConnections).length}/{cameraModelPairs.length}
                          </span>
                        </div>
                        <div className="stat-item">
                          <span className="stat-label">Total Alerts:</span>
                          <span className="stat-value">{alertCount}</span>
                        </div>
                        {criticalAlertCount > 0 && (
                          <div className="stat-item critical">
                            <span className="stat-label">Critical:</span>
                            <span className="stat-value">{criticalAlertCount}</span>
                          </div>
                        )}
                        
                        {/* Enhanced: Model breakdown in tracking (Updated for multiple instances) */}
                        {Object.keys(modelCounts).length > 1 && (
                          <div className="stat-item">
                            <span className="stat-label">Models Active:</span>
                            <span className="stat-value">{Object.keys(modelCounts).length}</span>
                          </div>
                        )}
                        
                        <div className="stat-item">
                          <span className="stat-label">Total Configurations:</span>
                          <span className="stat-value">{cameraModelPairs.length}</span>
                        </div>
                      </div>
                    )}
                  </div>
                </div>

                <div className="tracking-controls">
                  <button 
                    onClick={startTracking} 
                    disabled={isTracking || cameraModelPairs.length === 0} 
                    className="start-tracking-button"
                  >
                    <span className="button-icon">🚀</span>
                    Start Security Tracking
                  </button>
                  
                  <button 
                    onClick={stopTracking} 
                    disabled={!isTracking} 
                    className="stop-tracking-button"
                  >
                    <span className="button-icon">⏹️</span>
                    Stop Tracking
                  </button>
                </div>

                {isTracking && (
                  <div className="quick-actions-section">
                    <h4>Quick Actions</h4>
                    <div className="action-buttons">
                      <button 
                        onClick={onToggleDashboard}
                        className="action-button"
                      >
                        📊 Dashboard
                      </button>
                      
                      <button 
                        onClick={onToggleDetectionStatus}
                        className="action-button"
                      >
                        🔍 Status
                        {showDetectionStatus && <span className="active-dot">●</span>}
                      </button>
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Alerts Tab */}
            {activeTab === 'alerts' && (
              <div className="tab-panel">
                <h3>Alert Management</h3>
                
                <div className="alert-stats">
                  <div className="alert-stat-card">
                    <span className="stat-number">{alertCount}</span>
                    <span className="stat-label">Total Alerts</span>
                  </div>
                  
                  <div className="alert-stat-card critical">
                    <span className="stat-number">{criticalAlertCount}</span>
                    <span className="stat-label">Critical</span>
                  </div>
                </div>

                <div className="alert-actions">
                  <button 
                    onClick={onToggleAlerts}
                    className="alert-action-button primary"
                  >
                    🔔 Live Alerts
                    {alertCount > 0 && (
                      <span className="alert-badge">{alertCount}</span>
                    )}
                  </button>
                  
                  <button 
                    onClick={onToggleHistory}
                    className="alert-action-button"
                  >
                    📋 History
                  </button>
                </div>

                <div className="alert-settings">
                  <h4>Alert Settings</h4>
                  <div className="settings-list">
                    <div className="setting-item">
                      <label className="setting-label">
                        <input
                          type="checkbox"
                          checked={soundEnabled}
                          onChange={(e) => setSoundEnabled(e.target.checked)}
                        />
                        <span className="checkmark"></span>
                        Sound Alerts
                      </label>
                    </div>
                    
                    <div className="setting-item">
                      <label className="setting-label">
                        <input
                          type="checkbox"
                          checked={flashEnabled}
                          onChange={(e) => setFlashEnabled(e.target.checked)}
                        />
                        <span className="checkmark"></span>
                        Visual Flash
                      </label>
                    </div>
                  </div>
                </div>

                {!isTracking && (
                  <div className="alert-inactive-notice">
                    <span className="notice-icon">ℹ️</span>
                    <p>Alert system is inactive. Start tracking to begin receiving alerts.</p>
                  </div>
                )}
              </div>
            )}

            {/* Settings Tab */}
            {activeTab === 'settings' && (
              <div className="tab-panel">
                <h3>System Settings</h3>
                
                <div className="settings-section">
                  <h4>Recording Settings</h4>
                  <div className="setting-group">
                    <label>Output Directory:</label>
                    <div className="directory-input-group">
                      <input 
                        type="text" 
                        value={outputDir || "No directory selected"} 
                        readOnly 
                        className="directory-input"
                      />
                      <button 
                        onClick={selectOutputDirectory}
                        className="browse-button"
                      >
                        📁 Browse
                      </button>
                    </div>
                  </div>
                </div>

                <div className="settings-section">
                  <h4>System Information</h4>
                  <div className="system-info">
                    <div className="info-row">
                      <span className="info-label">Cameras Available:</span>
                      <span className="info-value">{cameras.length}</span>
                    </div>
                    <div className="info-row">
                      <span className="info-label">AI Models:</span>
                      <span className="info-value">{models.length}</span>
                    </div>
                    <div className="info-row">
                      <span className="info-label">Configured Pairs:</span>
                      <span className="info-value">{cameraModelPairs.length}</span>
                    </div>
                    <div className="info-row">
                      <span className="info-label">Status:</span>
                      <span className={`info-value ${systemStatus.status}`}>
                        {systemStatus.status.charAt(0).toUpperCase() + systemStatus.status.slice(1)}
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default ControlPanel;