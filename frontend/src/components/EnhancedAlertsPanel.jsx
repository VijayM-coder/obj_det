import React, { useState, useEffect, useRef } from 'react';
import './EnhancedAlertsPanel.css';

const EnhancedAlertsPanel = ({ cameras, isTrackingActive, alerts = [], onClearAlerts, onAcknowledgeAlert }) => {
  const [allAlerts, setAllAlerts] = useState([]);
  const [filteredAlerts, setFilteredAlerts] = useState([]);
  const [selectedAlert, setSelectedAlert] = useState(null);
  const [filters, setFilters] = useState({
    camera: 'all',
    severity: 'all',
    objectType: 'all',
    timeRange: 'all',
    status: 'all'
  });
  const [searchTerm, setSearchTerm] = useState('');
  const [sortBy, setSortBy] = useState('timestamp');
  const [sortOrder, setSortOrder] = useState('desc');
  const [isLoading, setIsLoading] = useState(false);
  const [viewMode, setViewMode] = useState('list'); // 'list' or 'grid'
  const [selectedAlerts, setSelectedAlerts] = useState(new Set());
  const [alertStats, setAlertStats] = useState({});
  
  const alertListRef = useRef(null);
  const detailPanelRef = useRef(null);

  // Use passed alerts prop instead of loading from services
  useEffect(() => {
    setIsLoading(true);
    
    // Use alerts passed as props, fallback to empty array
    const initialAlerts = alerts || [];
    
    // Sort by timestamp (newest first)
    const sortedAlerts = [...initialAlerts].sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
    setAllAlerts(sortedAlerts);
    setIsLoading(false);
    
    // Calculate initial stats
    calculateAlertStats(sortedAlerts);
  }, [alerts]);

  // Mock alert data for demonstration if no alerts provided
  useEffect(() => {
    if (!alerts || alerts.length === 0) {
      // Create some mock alerts for demonstration
      const mockAlerts = [
        {
          id: 'demo_1',
          cameraId: cameras[0]?.id || 'camera1',
          objectType: 'person',
          confidence: 0.92,
          timestamp: new Date(Date.now() - 300000).toISOString(), // 5 minutes ago
          severity: 'critical',
          bbox: [120, 80, 220, 180],
          imageData: null,
          acknowledged: false
        },
        {
          id: 'demo_2',
          cameraId: cameras[1]?.id || 'camera2',
          objectType: 'car',
          confidence: 0.78,
          timestamp: new Date(Date.now() - 600000).toISOString(), // 10 minutes ago
          severity: 'warning',
          bbox: [50, 100, 150, 200],
          imageData: null,
          acknowledged: false
        },
        {
          id: 'demo_3',
          cameraId: cameras[0]?.id || 'camera1',
          objectType: 'bicycle',
          confidence: 0.65,
          timestamp: new Date(Date.now() - 1200000).toISOString(), // 20 minutes ago
          severity: 'info',
          bbox: [80, 60, 140, 120],
          imageData: null,
          acknowledged: true
        }
      ];
      
      if (cameras.length > 0) {
        setAllAlerts(mockAlerts);
        calculateAlertStats(mockAlerts);
      }
    }
  }, [cameras, alerts]);

  // Calculate alert statistics
  const calculateAlertStats = (alerts) => {
    const stats = {
      total: alerts.length,
      critical: 0,
      warning: 0,
      info: 0,
      today: 0,
      thisWeek: 0,
      objectTypes: {},
      cameraBreakdown: {},
      hourlyDistribution: Array(24).fill(0)
    };

    const now = new Date();
    const todayStart = new Date(now.getFullYear(), now.getMonth(), now.getDate());
    const weekStart = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);

    alerts.forEach(alert => {
      // Severity counts
      const severity = alert.severity || 'info';
      stats[severity]++;

      // Time-based counts
      const alertDate = new Date(alert.timestamp);
      if (alertDate >= todayStart) stats.today++;
      if (alertDate >= weekStart) stats.thisWeek++;

      // Object type breakdown
      stats.objectTypes[alert.objectType] = (stats.objectTypes[alert.objectType] || 0) + 1;

      // Camera breakdown
      stats.cameraBreakdown[alert.cameraId] = (stats.cameraBreakdown[alert.cameraId] || 0) + 1;

      // Hourly distribution
      const hour = alertDate.getHours();
      stats.hourlyDistribution[hour]++;
    });

    setAlertStats(stats);
  };

  // Apply filters and search
  useEffect(() => {
    let filtered = [...allAlerts];

    // Camera filter
    if (filters.camera !== 'all') {
      filtered = filtered.filter(alert => alert.cameraId === filters.camera);
    }

    // Severity filter
    if (filters.severity !== 'all') {
      filtered = filtered.filter(alert => (alert.severity || 'info') === filters.severity);
    }

    // Object type filter
    if (filters.objectType !== 'all') {
      filtered = filtered.filter(alert => alert.objectType === filters.objectType);
    }

    // Time range filter
    const now = new Date();
    switch (filters.timeRange) {
      case 'today':
        const todayStart = new Date(now.getFullYear(), now.getMonth(), now.getDate());
        filtered = filtered.filter(alert => new Date(alert.timestamp) >= todayStart);
        break;
      case 'week':
        const weekStart = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
        filtered = filtered.filter(alert => new Date(alert.timestamp) >= weekStart);
        break;
      case 'month':
        const monthStart = new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000);
        filtered = filtered.filter(alert => new Date(alert.timestamp) >= monthStart);
        break;
    }

    // Search filter
    if (searchTerm) {
      filtered = filtered.filter(alert => 
        alert.objectType.toLowerCase().includes(searchTerm.toLowerCase()) ||
        getCameraName(alert.cameraId).toLowerCase().includes(searchTerm.toLowerCase()) ||
        (alert.id && alert.id.toLowerCase().includes(searchTerm.toLowerCase()))
      );
    }

    // Apply sorting
    filtered = sortAlerts(filtered, sortBy, sortOrder);

    setFilteredAlerts(filtered);
  }, [allAlerts, filters, searchTerm, sortBy, sortOrder]);

  // Sort alerts function
  const sortAlerts = (alerts, sortField, order) => {
    return [...alerts].sort((a, b) => {
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
        case 'severity':
          const severityOrder = { critical: 3, warning: 2, info: 1 };
          aVal = severityOrder[a.severity || 'info'];
          bVal = severityOrder[b.severity || 'info'];
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
  };

  // Get camera name by ID
  const getCameraName = (cameraId) => {
    const camera = cameras.find(cam => cam.id === cameraId);
    return camera ? (camera.name || `Camera ${cameraId}`) : `Camera ${cameraId}`;
  };

  // Get unique object types for filter
  const getUniqueObjectTypes = () => {
    return [...new Set(allAlerts.map(alert => alert.objectType))];
  };

  // Format timestamp for display
  const formatTimestamp = (isoString) => {
    try {
      const date = new Date(isoString);
      return {
        date: date.toLocaleDateString(),
        time: date.toLocaleTimeString(),
        relative: getRelativeTime(date)
      };
    } catch (e) {
      return { date: 'Invalid', time: 'Date', relative: 'Unknown' };
    }
  };

  // Get relative time
  const getRelativeTime = (date) => {
    const now = new Date();
    const diffMs = now - date;
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);

    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    if (diffDays < 7) return `${diffDays}d ago`;
    return date.toLocaleDateString();
  };

  // Handle filter change
  const handleFilterChange = (filterType, value) => {
    setFilters(prev => ({ ...prev, [filterType]: value }));
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

  // Handle alert selection
  const toggleAlertSelection = (alertId) => {
    setSelectedAlerts(prev => {
      const newSet = new Set(prev);
      if (newSet.has(alertId)) {
        newSet.delete(alertId);
      } else {
        newSet.add(alertId);
      }
      return newSet;
    });
  };

  // Select all filtered alerts
  const selectAllAlerts = () => {
    setSelectedAlerts(new Set(filteredAlerts.map(alert => alert.id)));
  };

  // Clear selection
  const clearSelection = () => {
    setSelectedAlerts(new Set());
  };

  // Handle alert acknowledgment
  const handleAcknowledgeAlert = (alertId) => {
    // Use callback if provided, otherwise update local state
    if (onAcknowledgeAlert) {
      onAcknowledgeAlert(alertId);
    } else {
      setAllAlerts(prev => 
        prev.map(alert => 
          alert.id === alertId ? { ...alert, acknowledged: true, acknowledgedAt: new Date().toISOString() } : alert
        )
      );
    }
  };

  // Clear selected alerts
  const handleClearAlerts = () => {
    if (window.confirm(`Are you sure you want to clear ${selectedAlerts.size} selected alerts?`)) {
      if (onClearAlerts) {
        // Use callback if provided
        const selectedAlertIds = Array.from(selectedAlerts);
        onClearAlerts(selectedAlertIds);
      } else {
        // Update local state
        setAllAlerts(prev => prev.filter(alert => !selectedAlerts.has(alert.id)));
      }
      setSelectedAlerts(new Set());
    }
  };

  // Get confidence color
  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.9) return '#ff3b30';
    if (confidence >= 0.7) return '#ff9500';
    return '#34c759';
  };

  // Get severity icon
  const getSeverityIcon = (severity) => {
    switch (severity) {
      case 'critical': return '🚨';
      case 'warning': return '⚠️';
      default: return '🔍';
    }
  };

  // Get sort icon
  const getSortIcon = (field) => {
    if (sortBy !== field) return '↕️';
    return sortOrder === 'asc' ? '↑' : '↓';
  };

  // Format bounding box
  const formatBoundingBox = (bbox) => {
    if (!bbox || !Array.isArray(bbox) || bbox.length !== 4) return 'N/A';
    const [x1, y1, x2, y2] = bbox.map(coord => Math.round(coord));
    const width = x2 - x1;
    const height = y2 - y1;
    return `Position: (${x1}, ${y1}), Size: ${width}×${height}px`;
  };

  return (
    <div className="enhanced-alerts-panel">
      {/* Header */}
      <div className="alerts-header">
        <div className="header-title">
          <h2>Alert Management</h2>
          <div className="tracking-status">
            {isTrackingActive ? (
              <span className="status-indicator active">🟢 Live Monitoring</span>
            ) : (
              <span className="status-indicator inactive">⭕ Monitoring Inactive</span>
            )}
          </div>
        </div>

        <div className="header-actions">
          <div className="view-mode-toggle">
            <button 
              className={`view-button ${viewMode === 'list' ? 'active' : ''}`}
              onClick={() => setViewMode('list')}
              title="List View"
            >
              📋
            </button>
            <button 
              className={`view-button ${viewMode === 'grid' ? 'active' : ''}`}
              onClick={() => setViewMode('grid')}
              title="Grid View"
            >
              ⊞
            </button>
          </div>
        </div>
      </div>

      {/* Statistics Dashboard */}
      <div className="alert-stats-dashboard">
        <div className="stats-grid">
          <div className="stat-card total">
            <span className="stat-number">{alertStats.total || 0}</span>
            <span className="stat-label">Total Alerts</span>
          </div>
          <div className="stat-card critical">
            <span className="stat-number">{alertStats.critical || 0}</span>
            <span className="stat-label">Critical</span>
          </div>
          <div className="stat-card warning">
            <span className="stat-number">{alertStats.warning || 0}</span>
            <span className="stat-label">Warning</span>
          </div>
          <div className="stat-card today">
            <span className="stat-number">{alertStats.today || 0}</span>
            <span className="stat-label">Today</span>
          </div>
        </div>

        {Object.keys(alertStats.objectTypes || {}).length > 0 && (
          <div className="top-detections">
            <h4>Top Detections</h4>
            <div className="detection-list">
              {Object.entries(alertStats.objectTypes)
                .sort(([,a], [,b]) => b - a)
                .slice(0, 5)
                .map(([type, count], index) => (
                  <div key={type} className="detection-item">
                    <span className="detection-rank">#{index + 1}</span>
                    <span className="detection-type">{type}</span>
                    <span className="detection-count">{count}</span>
                  </div>
                ))}
            </div>
          </div>
        )}
      </div>

      {/* Filters and Controls */}
      <div className="filters-section">
        <div className="filter-row">
          <div className="filter-group">
            <select 
              value={filters.camera} 
              onChange={(e) => handleFilterChange('camera', e.target.value)}
              className="filter-select"
            >
              <option value="all">All Cameras ({cameras.length})</option>
              {cameras.map(camera => (
                <option key={camera.id} value={camera.id}>
                  {camera.name || `Camera ${camera.id}`}
                  {alertStats.cameraBreakdown?.[camera.id] && 
                    ` (${alertStats.cameraBreakdown[camera.id]})`
                  }
                </option>
              ))}
            </select>
          </div>

          <div className="filter-group">
            <select 
              value={filters.severity} 
              onChange={(e) => handleFilterChange('severity', e.target.value)}
              className="filter-select"
            >
              <option value="all">All Severities</option>
              <option value="critical">🚨 Critical ({alertStats.critical || 0})</option>
              <option value="warning">⚠️ Warning ({alertStats.warning || 0})</option>
              <option value="info">🔍 Info ({alertStats.info || 0})</option>
            </select>
          </div>

          <div className="filter-group">
            <select 
              value={filters.objectType} 
              onChange={(e) => handleFilterChange('objectType', e.target.value)}
              className="filter-select"
            >
              <option value="all">All Objects</option>
              {getUniqueObjectTypes().map(type => (
                <option key={type} value={type}>
                  {type} ({alertStats.objectTypes?.[type] || 0})
                </option>
              ))}
            </select>
          </div>

          <div className="filter-group">
            <select 
              value={filters.timeRange} 
              onChange={(e) => handleFilterChange('timeRange', e.target.value)}
              className="filter-select"
            >
              <option value="all">All Time</option>
              <option value="today">Today ({alertStats.today || 0})</option>
              <option value="week">This Week ({alertStats.thisWeek || 0})</option>
              <option value="month">This Month</option>
            </select>
          </div>

          <div className="search-group">
            <input
              type="text"
              placeholder="Search alerts..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="search-input"
            />
          </div>
        </div>

        <div className="controls-row">
          <div className="selection-info">
            {selectedAlerts.size > 0 && (
              <span className="selection-count">
                {selectedAlerts.size} of {filteredAlerts.length} selected
              </span>
            )}
          </div>

          <div className="action-buttons">
            {selectedAlerts.size > 0 && (
              <>
                <button onClick={handleClearAlerts} className="action-button danger">
                  🗑️ Clear Selected
                </button>
                <button onClick={clearSelection} className="action-button">
                  ✕ Clear Selection
                </button>
              </>
            )}
            
            <button onClick={selectAllAlerts} className="action-button">
              ☑️ Select All
            </button>
          </div>

          <div className="sort-controls">
            <span>Sort by:</span>
            {['timestamp', 'severity', 'confidence', 'camera', 'objectType'].map(field => (
              <button
                key={field}
                className={`sort-button ${sortBy === field ? 'active' : ''}`}
                onClick={() => handleSortChange(field)}
              >
                {field.charAt(0).toUpperCase() + field.slice(1)} {getSortIcon(field)}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Main Content Area */}
      <div className="alerts-content">
        {/* Alert List */}
        <div className={`alerts-list-container ${selectedAlert ? 'with-detail' : ''}`}>
          <div className="results-info">
            <span>{filteredAlerts.length} alert{filteredAlerts.length !== 1 ? 's' : ''} found</span>
          </div>

          <div className={`alerts-list ${viewMode}`} ref={alertListRef}>
            {isLoading ? (
              <div className="loading-state">
                <div className="loading-spinner"></div>
                <p>Loading alerts...</p>
              </div>
            ) : filteredAlerts.length === 0 ? (
              <div className="no-alerts">
                <div className="no-alerts-icon">
                  {isTrackingActive ? '🔍' : '⏸️'}
                </div>
                <h3>
                  {isTrackingActive ? 'No alerts found' : 'Monitoring inactive'}
                </h3>
                <p>
                  {isTrackingActive 
                    ? 'Try adjusting your filters or wait for new detections'
                    : 'Start tracking to begin receiving alerts'
                  }
                </p>
              </div>
            ) : (
              filteredAlerts.map(alert => {
                const timestamp = formatTimestamp(alert.timestamp);
                const isSelected = selectedAlerts.has(alert.id);
                const isDetailSelected = selectedAlert?.id === alert.id;

                return (
                  <div 
                    key={alert.id} 
                    className={`alert-item ${alert.severity || 'info'} ${isSelected ? 'selected' : ''} ${isDetailSelected ? 'detail-selected' : ''}`}
                    onClick={() => setSelectedAlert(alert)}
                  >
                    <div className="alert-checkbox">
                      <input
                        type="checkbox"
                        checked={isSelected}
                        onChange={() => toggleAlertSelection(alert.id)}
                        onClick={(e) => e.stopPropagation()}
                      />
                    </div>

                    <div className="alert-content">
                      <div className="alert-header">
                        <div className="severity-indicator">
                          {getSeverityIcon(alert.severity)}
                        </div>
                        
                        <div className="alert-main-info">
                          <div className="alert-primary">
                            <span className="object-type">{alert.objectType}</span>
                            <span className="camera-name">{getCameraName(alert.cameraId)}</span>
                            <span 
                              className="confidence"
                              style={{ color: getConfidenceColor(alert.confidence) }}
                            >
                              {Math.round(alert.confidence * 100)}%
                            </span>
                          </div>
                          
                          <div className="alert-secondary">
                            <span className="timestamp">{timestamp.relative}</span>
                            <span className={`severity-badge ${alert.severity || 'info'}`}>
                              {(alert.severity || 'info').toUpperCase()}
                            </span>
                            {alert.acknowledged && (
                              <span className="acknowledged-badge">✓ Acknowledged</span>
                            )}
                          </div>
                        </div>
                      </div>

                      {viewMode === 'grid' && alert.imageData && (
                        <div className="alert-thumbnail">
                          <img 
                            src={`data:image/jpeg;base64,${alert.imageData}`} 
                            alt={`Alert: ${alert.objectType}`}
                            onError={(e) => e.target.style.display = 'none'}
                          />
                        </div>
                      )}
                    </div>

                    <div className="alert-actions">
                      {!alert.acknowledged && (
                        <button
                          className="acknowledge-button"
                          onClick={(e) => {
                            e.stopPropagation();
                            handleAcknowledgeAlert(alert.id);
                          }}
                          title="Acknowledge Alert"
                        >
                          ✓
                        </button>
                      )}
                      <button
                        className="detail-button"
                        onClick={(e) => {
                          e.stopPropagation();
                          setSelectedAlert(alert);
                        }}
                        title="View Details"
                      >
                        👁️
                      </button>
                    </div>
                  </div>
                );
              })
            )}
          </div>
        </div>

        {/* Alert Detail Panel */}
        {selectedAlert && (
          <div className="alert-detail-panel" ref={detailPanelRef}>
            <div className="detail-header">
              <h3>Alert Details</h3>
              <button 
                className="close-detail-button"
                onClick={() => setSelectedAlert(null)}
              >
                ✕
              </button>
            </div>

            <div className="detail-content">
              {/* Alert Image */}
              {selectedAlert.imageData && (
                <div className="detail-image-container">
                  <img 
                    src={`data:image/jpeg;base64,${selectedAlert.imageData}`} 
                    alt={`Alert: ${selectedAlert.objectType}`}
                    className="detail-image"
                  />
                  
                  {/* Bounding Box Overlay */}
                  {selectedAlert.bbox && (
                    <div className="detail-bbox-overlay">
                      <div 
                        className="detail-bounding-box"
                        style={{
                          left: `${(selectedAlert.bbox[0] / 640) * 100}%`,
                          top: `${(selectedAlert.bbox[1] / 480) * 100}%`,
                          width: `${((selectedAlert.bbox[2] - selectedAlert.bbox[0]) / 640) * 100}%`,
                          height: `${((selectedAlert.bbox[3] - selectedAlert.bbox[1]) / 480) * 100}%`
                        }}
                      >
                        <div className="detail-bbox-label">
                          {selectedAlert.objectType} ({Math.round(selectedAlert.confidence * 100)}%)
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* Alert Information */}
              <div className="detail-info">
                <div className="info-section">
                  <h4>Detection Information</h4>
                  <div className="info-grid">
                    <div className="info-item">
                      <span className="info-label">Object Detected:</span>
                      <span className="info-value">{selectedAlert.objectType}</span>
                    </div>
                    <div className="info-item">
                      <span className="info-label">Confidence:</span>
                      <span 
                        className="info-value"
                        style={{ color: getConfidenceColor(selectedAlert.confidence) }}
                      >
                        {Math.round(selectedAlert.confidence * 100)}%
                      </span>
                    </div>
                    <div className="info-item">
                      <span className="info-label">Severity:</span>
                      <span className={`severity-badge ${selectedAlert.severity || 'info'}`}>
                        {getSeverityIcon(selectedAlert.severity)} {(selectedAlert.severity || 'info').toUpperCase()}
                      </span>
                    </div>
                  </div>
                </div>

                <div className="info-section">
                  <h4>Location & Time</h4>
                  <div className="info-grid">
                    <div className="info-item">
                      <span className="info-label">Camera:</span>
                      <span className="info-value">{getCameraName(selectedAlert.cameraId)}</span>
                    </div>
                    <div className="info-item">
                      <span className="info-label">Detection Time:</span>
                      <span className="info-value">
                        {formatTimestamp(selectedAlert.timestamp).date} at {formatTimestamp(selectedAlert.timestamp).time}
                      </span>
                    </div>
                    <div className="info-item">
                      <span className="info-label">Position:</span>
                      <span className="info-value">{formatBoundingBox(selectedAlert.bbox)}</span>
                    </div>
                  </div>
                </div>

                <div className="info-section">
                  <h4>Technical Details</h4>
                  <div className="info-grid">
                    <div className="info-item">
                      <span className="info-label">Alert ID:</span>
                      <span className="info-value alert-id">{selectedAlert.id}</span>
                    </div>
                    {selectedAlert.acknowledged && (
                      <div className="info-item">
                        <span className="info-label">Acknowledged:</span>
                        <span className="info-value">
                          {formatTimestamp(selectedAlert.acknowledgedAt).date} at {formatTimestamp(selectedAlert.acknowledgedAt).time}
                        </span>
                      </div>
                    )}
                  </div>
                </div>
              </div>

              {/* Actions */}
              <div className="detail-actions">
                {!selectedAlert.acknowledged && (
                  <button
                    className="detail-action-button primary"
                    onClick={() => handleAcknowledgeAlert(selectedAlert.id)}
                  >
                    ✓ Acknowledge Alert
                  </button>
                )}
                
                <button
                  className="detail-action-button secondary"
                  onClick={() => {
                    const alertIds = new Set([selectedAlert.id]);
                    setSelectedAlerts(alertIds);
                  }}
                >
                  ☑️ Select for Actions
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default EnhancedAlertsPanel;