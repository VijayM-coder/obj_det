// components/AlertHistory.jsx - Comprehensive alert history and analytics
import React, { useState, useEffect } from 'react';
import { getRecentAlerts, clearAlerts } from '../services/alertServices';
import './AlertHistory.css';

const AlertHistory = ({ cameras, isTrackingActive }) => {
  const [allAlerts, setAllAlerts] = useState([]);
  const [filteredAlerts, setFilteredAlerts] = useState([]);
  const [filters, setFilters] = useState({
    camera: 'all',
    severity: 'all',
    objectType: 'all',
    dateRange: 'today',
    status: 'all'
  });
  const [searchTerm, setSearchTerm] = useState('');
  const [sortBy, setSortBy] = useState('timestamp');
  const [sortOrder, setSortOrder] = useState('desc');
  const [selectedAlerts, setSelectedAlerts] = useState(new Set());
  const [analytics, setAnalytics] = useState({});
  const [isLoading, setIsLoading] = useState(true);

  // Load alert history
  useEffect(() => {
    setIsLoading(true);
    
    let allHistoryAlerts = [];
    cameras.forEach(camera => {
      const cameraAlerts = getRecentAlerts(camera.id, 100); // Get more history
      allHistoryAlerts = [...allHistoryAlerts, ...cameraAlerts];
    });
    
    // Sort by timestamp
    allHistoryAlerts.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
    
    setAllAlerts(allHistoryAlerts);
    setIsLoading(false);
  }, [cameras]);

  // Calculate analytics
  useEffect(() => {
    if (allAlerts.length > 0) {
      const analytics = calculateAnalytics(allAlerts);
      setAnalytics(analytics);
    }
  }, [allAlerts]);

  // Apply filters
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

    // Date range filter
    const now = new Date();
    switch (filters.dateRange) {
      case 'today':
        const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
        filtered = filtered.filter(alert => new Date(alert.timestamp) >= today);
        break;
      case 'week':
        const weekAgo = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
        filtered = filtered.filter(alert => new Date(alert.timestamp) >= weekAgo);
        break;
      case 'month':
        const monthAgo = new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000);
        filtered = filtered.filter(alert => new Date(alert.timestamp) >= monthAgo);
        break;
    }

    // Search filter
    if (searchTerm) {
      filtered = filtered.filter(alert => 
        alert.objectType.toLowerCase().includes(searchTerm.toLowerCase()) ||
        getCameraName(alert.cameraId).toLowerCase().includes(searchTerm.toLowerCase())
      );
    }

    // Apply sorting
    filtered = sortAlerts(filtered, sortBy, sortOrder);

    setFilteredAlerts(filtered);
  }, [allAlerts, filters, searchTerm, sortBy, sortOrder]);

  // Calculate analytics
  const calculateAnalytics = (alerts) => {
    const totalAlerts = alerts.length;
    
    // Group by object type
    const objectTypes = alerts.reduce((acc, alert) => {
      acc[alert.objectType] = (acc[alert.objectType] || 0) + 1;
      return acc;
    }, {});

    // Group by camera
    const cameraStats = alerts.reduce((acc, alert) => {
      acc[alert.cameraId] = (acc[alert.cameraId] || 0) + 1;
      return acc;
    }, {});

    // Group by severity
    const severityStats = alerts.reduce((acc, alert) => {
      const severity = alert.severity || 'info';
      acc[severity] = (acc[severity] || 0) + 1;
      return acc;
    }, {});

    // Average confidence
    const avgConfidence = alerts.length > 0 
      ? alerts.reduce((sum, alert) => sum + alert.confidence, 0) / alerts.length
      : 0;

    // Peak hours
    const hourlyStats = alerts.reduce((acc, alert) => {
      const hour = new Date(alert.timestamp).getHours();
      acc[hour] = (acc[hour] || 0) + 1;
      return acc;
    }, {});

    const peakHour = Object.entries(hourlyStats).reduce((a, b) => 
      hourlyStats[a[0]] > hourlyStats[b[0]] ? a : b, [0, 0]
    );

    return {
      totalAlerts,
      objectTypes,
      cameraStats,
      severityStats,
      avgConfidence,
      peakHour: peakHour[0],
      hourlyStats
    };
  };

  // Sort alerts
  const sortAlerts = (alerts, field, order) => {
    return [...alerts].sort((a, b) => {
      let aVal, bVal;
      
      switch (field) {
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

  // Get unique object types for filter
  const getUniqueObjectTypes = () => {
    return [...new Set(allAlerts.map(alert => alert.objectType))];
  };

  // Get camera name
  const getCameraName = (cameraId) => {
    const camera = cameras.find(cam => cam.id === cameraId);
    return camera ? (camera.name || `Camera ${cameraId}`) : `Camera ${cameraId}`;
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

  // Select all alerts
  const selectAllAlerts = () => {
    setSelectedAlerts(new Set(filteredAlerts.map(alert => alert.id)));
  };

  // Clear selection
  const clearSelection = () => {
    setSelectedAlerts(new Set());
  };

  // Export selected alerts
  const exportAlerts = () => {
    const selectedAlertData = filteredAlerts.filter(alert => 
      selectedAlerts.has(alert.id)
    );
    
    const csvContent = convertToCSV(selectedAlertData);
    downloadCSV(csvContent, 'alert-history.csv');
  };

  // Convert to CSV
  const convertToCSV = (alerts) => {
    const headers = ['Timestamp', 'Camera', 'Object Type', 'Confidence', 'Severity'];
    const rows = alerts.map(alert => [
      alert.timestamp,
      getCameraName(alert.cameraId),
      alert.objectType,
      Math.round(alert.confidence * 100) + '%',
      alert.severity || 'info'
    ]);
    
    return [headers, ...rows].map(row => row.join(',')).join('\n');
  };

  // Download CSV
  const downloadCSV = (content, filename) => {
    const blob = new Blob([content], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    window.URL.revokeObjectURL(url);
  };

  // Get severity icon
  const getSeverityIcon = (severity) => {
    switch (severity) {
      case 'critical': return '🔴';
      case 'warning': return '🟡';
      default: return '🟢';
    }
  };

  // Get sort icon
  const getSortIcon = (field) => {
    if (sortBy !== field) return '↕️';
    return sortOrder === 'asc' ? '↑' : '↓';
  };

  // Format timestamp
  const formatTimestamp = (timestamp) => {
    try {
      const date = new Date(timestamp);
      return {
        date: date.toLocaleDateString(),
        time: date.toLocaleTimeString()
      };
    } catch {
      return { date: 'Invalid', time: 'Date' };
    }
  };

  return (
    <div className="alert-history">
      {/* Header */}
      <div className="history-header">
        <h3>Alert History & Analytics</h3>
        <div className="header-actions">
          {selectedAlerts.size > 0 && (
            <>
              <span className="selection-count">
                {selectedAlerts.size} selected
              </span>
              <button className="export-button" onClick={exportAlerts}>
                📁 Export CSV
              </button>
              <button className="clear-selection-button" onClick={clearSelection}>
                Clear Selection
              </button>
            </>
          )}
        </div>
      </div>

      {/* Analytics Summary */}
      {analytics.totalAlerts > 0 && (
        <div className="analytics-summary">
          <div className="summary-cards">
            <div className="summary-card">
              <span className="card-value">{analytics.totalAlerts}</span>
              <span className="card-label">Total Alerts</span>
            </div>
            <div className="summary-card">
              <span className="card-value">
                {Math.round(analytics.avgConfidence * 100)}%
              </span>
              <span className="card-label">Avg Confidence</span>
            </div>
            <div className="summary-card">
              <span className="card-value">{analytics.peakHour}:00</span>
              <span className="card-label">Peak Hour</span>
            </div>
            <div className="summary-card">
              <span className="card-value">
                {Object.keys(analytics.objectTypes).length}
              </span>
              <span className="card-label">Object Types</span>
            </div>
          </div>

          {/* Top detections */}
          <div className="top-detections">
            <div className="detection-category">
              <h5>Top Object Types</h5>
              {Object.entries(analytics.objectTypes)
                .sort(([,a], [,b]) => b - a)
                .slice(0, 3)
                .map(([type, count]) => (
                  <div key={type} className="detection-item">
                    <span className="detection-type">{type}</span>
                    <span className="detection-count">{count}</span>
                  </div>
                ))}
            </div>

            <div className="detection-category">
              <h5>Most Active Cameras</h5>
              {Object.entries(analytics.cameraStats)
                .sort(([,a], [,b]) => b - a)
                .slice(0, 3)
                .map(([cameraId, count]) => (
                  <div key={cameraId} className="detection-item">
                    <span className="detection-type">{getCameraName(cameraId)}</span>
                    <span className="detection-count">{count}</span>
                  </div>
                ))}
            </div>
          </div>
        </div>
      )}

      {/* Filters */}
      <div className="filters-section">
        <div className="filter-row">
          <div className="filter-group">
            <label>Camera:</label>
            <select 
              value={filters.camera} 
              onChange={(e) => handleFilterChange('camera', e.target.value)}
            >
              <option value="all">All Cameras</option>
              {cameras.map(camera => (
                <option key={camera.id} value={camera.id}>
                  {camera.name || `Camera ${camera.id}`}
                </option>
              ))}
            </select>
          </div>

          <div className="filter-group">
            <label>Severity:</label>
            <select 
              value={filters.severity} 
              onChange={(e) => handleFilterChange('severity', e.target.value)}
            >
              <option value="all">All Severities</option>
              <option value="critical">Critical</option>
              <option value="warning">Warning</option>
              <option value="info">Info</option>
            </select>
          </div>

          <div className="filter-group">
            <label>Object Type:</label>
            <select 
              value={filters.objectType} 
              onChange={(e) => handleFilterChange('objectType', e.target.value)}
            >
              <option value="all">All Types</option>
              {getUniqueObjectTypes().map(type => (
                <option key={type} value={type}>{type}</option>
              ))}
            </select>
          </div>

          <div className="filter-group">
            <label>Date Range:</label>
            <select 
              value={filters.dateRange} 
              onChange={(e) => handleFilterChange('dateRange', e.target.value)}
            >
              <option value="today">Today</option>
              <option value="week">This Week</option>
              <option value="month">This Month</option>
              <option value="all">All Time</option>
            </select>
          </div>

          <div className="filter-group">
            <input
              type="text"
              placeholder="Search alerts..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="search-input"
            />
          </div>
        </div>
      </div>

      {/* Results count and actions */}
      <div className="results-header">
        <div className="results-info">
          <span>{filteredAlerts.length} alert{filteredAlerts.length !== 1 ? 's' : ''} found</span>
          <button onClick={selectAllAlerts} className="select-all-button">
            Select All
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

      {/* Alert list */}
      <div className="alert-list">
        {isLoading ? (
          <div className="loading-state">
            <div className="loading-spinner"></div>
            <p>Loading alert history...</p>
          </div>
        ) : filteredAlerts.length === 0 ? (
          <div className="no-alerts">
            <div className="no-alerts-icon">📋</div>
            <p>No alerts found matching the current filters</p>
          </div>
        ) : (
          filteredAlerts.map(alert => {
            const timestamp = formatTimestamp(alert.timestamp);
            const isSelected = selectedAlerts.has(alert.id);
            
            return (
              <div 
                key={alert.id} 
                className={`alert-history-item ${alert.severity || 'info'} ${isSelected ? 'selected' : ''}`}
                onClick={() => toggleAlertSelection(alert.id)}
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
                  <div className="alert-main-info">
                    <div className="severity-indicator">
                      {getSeverityIcon(alert.severity)}
                    </div>
                    
                    <div className="alert-details">
                      <div className="alert-primary">
                        <span className="object-type">{alert.objectType}</span>
                        <span className="camera-name">{getCameraName(alert.cameraId)}</span>
                        <span className="confidence">
                          {Math.round(alert.confidence * 100)}%
                        </span>
                      </div>
                      
                      <div className="alert-secondary">
                        <span className="timestamp">
                          {timestamp.date} at {timestamp.time}
                        </span>
                        <span className={`severity-badge ${alert.severity || 'info'}`}>
                          {(alert.severity || 'info').toUpperCase()}
                        </span>
                      </div>
                    </div>
                  </div>

                  {alert.imageData && (
                    <div className="alert-thumbnail">
                      <img 
                        src={`data:image/jpeg;base64,${alert.imageData}`} 
                        alt={`Alert: ${alert.objectType}`}
                        onError={(e) => e.target.style.display = 'none'}
                      />
                    </div>
                  )}
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
};

export default AlertHistory;