import React, { useState, useEffect } from 'react';
import { getRecentAlerts, clearAlerts, subscribeToAlerts } from '../services/alertServices';
import './AlertsPanel.css';

const AlertsPanel = ({ cameras, isTrackingActive }) => {
  const [allAlerts, setAllAlerts] = useState([]);
  const [filteredAlerts, setFilteredAlerts] = useState([]);
  const [selectedCamera, setSelectedCamera] = useState('all');
  const [searchTerm, setSearchTerm] = useState('');
  const [isLoading, setIsLoading] = useState(true);
  const [sortBy, setSortBy] = useState('timestamp'); // timestamp, confidence, camera
  const [sortOrder, setSortOrder] = useState('desc'); // asc, desc
  
  // Load initial alerts and subscribe to new ones
  useEffect(() => {
    setIsLoading(true);
    
    // Get all recent alerts for all cameras
    let initialAlerts = [];
    cameras.forEach(camera => {
      const cameraAlerts = getRecentAlerts(camera.id);
      initialAlerts = [...initialAlerts, ...cameraAlerts];
    });
    
    // Sort by timestamp (newest first)
    initialAlerts.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
    setAllAlerts(initialAlerts);
    setIsLoading(false);
    
    // Only set up subscriptions if tracking is active
    if (!isTrackingActive) {
      return;
    }

    console.log('AlertsPanel: Setting up alert subscriptions for tracking mode');
    
    // Set up alert subscription for all cameras
    const unsubscribe = subscribeToAlerts('all', (newAlert) => {
      console.log('AlertsPanel: Received new alert:', newAlert);
      
      setAllAlerts(prev => {
        const updated = [newAlert, ...prev];
        // Sort by current sort settings
        return sortAlerts(updated, sortBy, sortOrder);
      });
    });
    
    // Clean up subscriptions on unmount or when tracking stops
    return () => {
      console.log('AlertsPanel: Cleaning up alert subscriptions');
      unsubscribe();
    };
  }, [cameras, isTrackingActive, sortBy, sortOrder]);
  
  // Sort alerts function
  const sortAlerts = (alerts, sortField, order) => {
    const sorted = [...alerts].sort((a, b) => {
      let aVal, bVal;
      
      switch (sortField) {
        case 'confidence':
          aVal = a.confidence;
          bVal = b.confidence;
          break;
        case 'camera':
          aVal = getCameraName(a.cameraId);
          bVal = getCameraName(b.cameraId);
          break;
        case 'objectType':
          aVal = a.objectType;
          bVal = b.objectType;
          break;
        case 'timestamp':
        default:
          aVal = new Date(a.timestamp);
          bVal = new Date(b.timestamp);
          break;
      }
      
      if (order === 'asc') {
        return aVal > bVal ? 1 : aVal < bVal ? -1 : 0;
      } else {
        return aVal < bVal ? 1 : aVal > bVal ? -1 : 0;
      }
    });
    
    return sorted.slice(0, 100); // Limit to 100 alerts
  };
  
  // Update filtered alerts when selection changes
  useEffect(() => {
    let filtered = [...allAlerts];
    
    // Filter by camera
    if (selectedCamera !== 'all') {
      filtered = filtered.filter(alert => alert.cameraId === selectedCamera);
    }
    
    // Filter by search term
    if (searchTerm) {
      filtered = filtered.filter(alert => 
        alert.objectType.toLowerCase().includes(searchTerm.toLowerCase())
      );
    }
    
    // Apply sorting
    filtered = sortAlerts(filtered, sortBy, sortOrder);
    
    setFilteredAlerts(filtered);
  }, [allAlerts, selectedCamera, searchTerm, sortBy, sortOrder]);
  
  // Format timestamp for display
  const formatTimestamp = (isoString) => {
    try {
      const date = new Date(isoString);
      return {
        date: date.toLocaleDateString(),
        time: date.toLocaleTimeString()
      };
    } catch (e) {
      return { date: 'Invalid', time: 'Date' };
    }
  };
  
  // Get camera name by ID
  const getCameraName = (cameraId) => {
    const camera = cameras.find(cam => cam.id === cameraId);
    return camera ? (camera.name || `Camera ${cameraId}`) : `Camera ${cameraId}`;
  };
  
  // Handle camera selection change
  const handleCameraChange = (e) => {
    setSelectedCamera(e.target.value);
  };
  
  // Handle search input change
  const handleSearchChange = (e) => {
    setSearchTerm(e.target.value);
  };
  
  // Handle sort change
  const handleSortChange = (field) => {
    if (sortBy === field) {
      setSortOrder(prev => prev === 'asc' ? 'desc' : 'asc');
    } else {
      setSortBy(field);
      setSortOrder('desc');
    }
  };
  
  // Clear alerts for selected camera
  const handleClearAlerts = () => {
    if (window.confirm('Are you sure you want to clear these alerts?')) {
      if (selectedCamera === 'all') {
        cameras.forEach(camera => clearAlerts(camera.id));
        setAllAlerts([]);
      } else {
        clearAlerts(selectedCamera);
        setAllAlerts(prev => prev.filter(alert => alert.cameraId !== selectedCamera));
      }
    }
  };
  
  // Get confidence color
  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.9) return '#ff4444';
    if (confidence >= 0.7) return '#ff8800';
    return '#44ff44';
  };
  
  // Get sort icon
  const getSortIcon = (field) => {
    if (sortBy !== field) return '↕️';
    return sortOrder === 'asc' ? '↑' : '↓';
  };
  
  return (
    <div className="alerts-panel">
      <div className="alerts-header">
        <div className="header-title">
          <h2>Alert History</h2>
          {!isTrackingActive && (
            <div className="tracking-status">
              <span className="status-indicator inactive">Tracking Inactive</span>
            </div>
          )}
          {isTrackingActive && (
            <div className="tracking-status">
              <span className="status-indicator active">Live Monitoring</span>
            </div>
          )}
        </div>
        
        <div className="filter-controls">
          <div className="filter-row">
            <div className="camera-filter">
              <label>Camera:</label>
              <select value={selectedCamera} onChange={handleCameraChange}>
                <option value="all">All Cameras ({cameras.length})</option>
                {cameras.map(camera => (
                  <option key={camera.id} value={camera.id}>
                    {camera.name || `Camera ${camera.id}`}
                  </option>
                ))}
              </select>
            </div>
            
            <div className="search-filter">
              <input 
                type="text" 
                placeholder="Search by object type..." 
                value={searchTerm} 
                onChange={handleSearchChange}
              />
            </div>
            
            <button 
              className="clear-button" 
              onClick={handleClearAlerts}
              disabled={filteredAlerts.length === 0}
            >
              Clear {selectedCamera === 'all' ? 'All' : 'Selected'}
            </button>
          </div>
          
          <div className="sort-controls">
            <span>Sort by:</span>
            <button 
              className={`sort-button ${sortBy === 'timestamp' ? 'active' : ''}`}
              onClick={() => handleSortChange('timestamp')}
            >
              Time {getSortIcon('timestamp')}
            </button>
            <button 
              className={`sort-button ${sortBy === 'confidence' ? 'active' : ''}`}
              onClick={() => handleSortChange('confidence')}
            >
              Confidence {getSortIcon('confidence')}
            </button>
            <button 
              className={`sort-button ${sortBy === 'camera' ? 'active' : ''}`}
              onClick={() => handleSortChange('camera')}
            >
              Camera {getSortIcon('camera')}
            </button>
            <button 
              className={`sort-button ${sortBy === 'objectType' ? 'active' : ''}`}
              onClick={() => handleSortChange('objectType')}
            >
              Object {getSortIcon('objectType')}
            </button>
          </div>
        </div>
      </div>
      
      <div className="alerts-stats">
        <div className="stat-item">
          <span className="stat-value">{filteredAlerts.length}</span>
          <span className="stat-label">Alerts</span>
        </div>
        {filteredAlerts.length > 0 && (
          <>
            <div className="stat-item">
              <span className="stat-value">
                {Math.round(filteredAlerts.reduce((sum, alert) => sum + alert.confidence, 0) / filteredAlerts.length * 100)}%
              </span>
              <span className="stat-label">Avg Confidence</span>
            </div>
            <div className="stat-item">
              <span className="stat-value">
                {new Set(filteredAlerts.map(a => a.cameraId)).size}
              </span>
              <span className="stat-label">Cameras</span>
            </div>
          </>
        )}
      </div>
      
      <div className="alerts-list">
        {isLoading ? (
          <div className="loading-state">
            <div className="loading-spinner"></div>
            <p>Loading alerts...</p>
          </div>
        ) : filteredAlerts.length === 0 ? (
          <div className="no-alerts">
            {isTrackingActive ? (
              <div className="no-alerts-content">
                <div className="no-alerts-icon">🔍</div>
                <p>No alerts found</p>
                <small>Alerts will appear here when objects are detected</small>
              </div>
            ) : (
              <div className="no-alerts-content">
                <div className="no-alerts-icon">⏸️</div>
                <p>Tracking is not active</p>
                <small>Start tracking to begin receiving alerts</small>
              </div>
            )}
          </div>
        ) : (
          filteredAlerts.map(alert => {
            const timestamp = formatTimestamp(alert.timestamp);
            return (
              <div key={alert.id} className="alert-card">
                <div className="alert-header">
                  <div className="alert-info">
                    <span className="alert-camera">{getCameraName(alert.cameraId)}</span>
                    <div className="alert-time">
                      <span className="alert-date">{timestamp.date}</span>
                      <span className="alert-time-value">{timestamp.time}</span>
                    </div>
                  </div>
                  <div className="alert-classification">
                    <span className="alert-type">{alert.objectType}</span>
                    <span 
                      className="alert-confidence"
                      style={{ color: getConfidenceColor(alert.confidence) }}
                    >
                      {Math.round(alert.confidence * 100)}%
                    </span>
                  </div>
                </div>
                
                {alert.imageData && (
                  <div className="alert-image">
                    <img 
                      src={`data:image/jpeg;base64,${alert.imageData}`} 
                      alt={`Alert: ${alert.objectType}`}
                      onError={(e) => {
                        e.target.parentElement.style.display = 'none';
                      }}
                    />
                  </div>
                )}
                
                {alert.bbox && (
                  <div className="alert-details">
                    <small>
                      Detection box: {alert.bbox.map(coord => Math.round(coord)).join(', ')}
                    </small>
                  </div>
                )}
              </div>
            );
          })
        )}
      </div>
    </div>
  );
};

export default AlertsPanel;