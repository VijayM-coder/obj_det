// components/SecurityDashboard.jsx - Real-time security overview
import React, { useState, useEffect } from 'react';
import { getRecentAlerts } from '../services/alertServices';
import './SecurityDashboard.css';

const SecurityDashboard = ({ 
  isTrackingActive, 
  trackingPairs, 
  cameras, 
  alertCount, 
  criticalAlertCount,
  activeConnections 
}) => {
  const [recentAlerts, setRecentAlerts] = useState([]);
  const [systemStats, setSystemStats] = useState({
    uptime: 0,
    totalDetections: 0,
    avgConfidence: 0,
    threatLevel: 'low'
  });
  const [zoneStatus, setZoneStatus] = useState({});

  // Define security zones (you can customize this based on your setup)
  const securityZones = [
    { id: 'north', name: 'North Sector', cameras: [] },
    { id: 'south', name: 'South Sector', cameras: [] },
    { id: 'east', name: 'East Sector', cameras: [] },
    { id: 'west', name: 'West Sector', cameras: [] }
  ];

  useEffect(() => {
    if (isTrackingActive) {
      // Load recent alerts from all cameras
      let allRecentAlerts = [];
      cameras.forEach(camera => {
        const cameraAlerts = getRecentAlerts(camera.id, 10);
        allRecentAlerts = [...allRecentAlerts, ...cameraAlerts];
      });
      
      // Sort by timestamp (newest first)
      allRecentAlerts.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
      setRecentAlerts(allRecentAlerts.slice(0, 5)); // Show only 5 most recent

      // Calculate system stats
      const totalDetections = allRecentAlerts.length;
      const avgConfidence = totalDetections > 0 
        ? allRecentAlerts.reduce((sum, alert) => sum + alert.confidence, 0) / totalDetections
        : 0;

      // Determine threat level based on critical alerts
      let threatLevel = 'low';
      if (criticalAlertCount > 0) {
        threatLevel = criticalAlertCount >= 3 ? 'high' : 'medium';
      }

      setSystemStats(prev => ({
        ...prev,
        totalDetections,
        avgConfidence,
        threatLevel
      }));
    }
  }, [isTrackingActive, cameras, criticalAlertCount]);

  // Update system uptime
  useEffect(() => {
    if (isTrackingActive) {
      const startTime = Date.now();
      const interval = setInterval(() => {
        const uptime = Math.floor((Date.now() - startTime) / 1000);
        setSystemStats(prev => ({ ...prev, uptime }));
      }, 1000);

      return () => clearInterval(interval);
    }
  }, [isTrackingActive]);

  // Format uptime
  const formatUptime = (seconds) => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  // Get threat level color
  const getThreatLevelColor = (level) => {
    switch (level) {
      case 'high': return '#ff3b30';
      case 'medium': return '#ff9500';
      default: return '#34c759';
    }
  };

  // Get camera status
  const getCameraStatus = (cameraId) => {
    const camera = cameras.find(c => c.id === cameraId);
    return camera?.status || 'unknown';
  };

  // Format timestamp
  const formatTime = (timestamp) => {
    try {
      return new Date(timestamp).toLocaleTimeString();
    } catch {
      return 'Invalid time';
    }
  };

  // Get severity icon
  const getSeverityIcon = (severity) => {
    switch (severity) {
      case 'critical': return '🔴';
      case 'warning': return '🟡';
      default: return '🟢';
    }
  };

  if (!isTrackingActive) {
    return (
      <div className="security-dashboard inactive">
        <div className="dashboard-header">
          <h3>Security Dashboard</h3>
          <span className="status-badge inactive">System Offline</span>
        </div>
        <p className="inactive-message">Start tracking to activate security monitoring</p>
      </div>
    );
  }

  return (
    <div className="security-dashboard active">
      {/* Dashboard Header */}
      <div className="dashboard-header">
        <h3>Security Dashboard</h3>
        <div className="system-status">
          <span className="status-badge active">System Active</span>
          <span className="uptime">Uptime: {formatUptime(systemStats.uptime)}</span>
        </div>
      </div>

      {/* Threat Level Indicator */}
      <div className="threat-level-section">
        <div 
          className={`threat-level ${systemStats.threatLevel}`}
          style={{ borderColor: getThreatLevelColor(systemStats.threatLevel) }}
        >
          <div className="threat-indicator">
            <div 
              className="threat-bar"
              style={{ backgroundColor: getThreatLevelColor(systemStats.threatLevel) }}
            />
            <span className="threat-label">
              Threat Level: {systemStats.threatLevel.toUpperCase()}
            </span>
          </div>
          
          {criticalAlertCount > 0 && (
            <div className="critical-warning">
              <span className="warning-icon">⚠️</span>
              <span>{criticalAlertCount} Critical Alert{criticalAlertCount > 1 ? 's' : ''} Active</span>
            </div>
          )}
        </div>
      </div>

      {/* System Metrics */}
      <div className="metrics-grid">
        <div className="metric-card">
          <div className="metric-icon">📹</div>
          <div className="metric-content">
            <span className="metric-value">{activeConnections}/{trackingPairs.length}</span>
            <span className="metric-label">Active Cameras</span>
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-icon">🎯</div>
          <div className="metric-content">
            <span className="metric-value">{systemStats.totalDetections}</span>
            <span className="metric-label">Total Detections</span>
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-icon">📊</div>
          <div className="metric-content">
            <span className="metric-value">
              {Math.round(systemStats.avgConfidence * 100)}%
            </span>
            <span className="metric-label">Avg Confidence</span>
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-icon">🚨</div>
          <div className="metric-content">
            <span className="metric-value">{alertCount}</span>
            <span className="metric-label">Recent Alerts</span>
          </div>
        </div>
      </div>

      {/* Camera Status Grid */}
      <div className="camera-status-section">
        <h4>Camera Status</h4>
        <div className="camera-status-grid">
          {trackingPairs.map(pair => {
            const status = getCameraStatus(pair.cameraId);
            const camera = cameras.find(c => c.id === pair.cameraId);
            
            return (
              <div key={pair.cameraId} className={`camera-status-card ${status}`}>
                <div className="camera-info">
                  <span className="camera-name">
                    {camera?.name || `Camera ${pair.cameraId}`}
                  </span>
                  <span className={`status-indicator ${status}`}>
                    <span className="status-dot"></span>
                    {status.charAt(0).toUpperCase() + status.slice(1)}
                  </span>
                </div>
                <div className="camera-details">
                  <span className="model-info">{pair.model}</span>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Recent Activity Feed */}
      <div className="activity-feed-section">
        <h4>Recent Activity</h4>
        <div className="activity-feed">
          {recentAlerts.length === 0 ? (
            <div className="no-activity">
              <span className="activity-icon">✅</span>
              <span>All quiet - no recent detections</span>
            </div>
          ) : (
            recentAlerts.map(alert => {
              const camera = cameras.find(c => c.id === alert.cameraId);
              return (
                <div key={alert.id} className={`activity-item ${alert.severity || 'info'}`}>
                  <div className="activity-icon">
                    {getSeverityIcon(alert.severity)}
                  </div>
                  <div className="activity-content">
                    <div className="activity-main">
                      <span className="object-type">{alert.objectType}</span>
                      <span className="camera-name">
                        {camera?.name || `Camera ${alert.cameraId}`}
                      </span>
                    </div>
                    <div className="activity-meta">
                      <span className="confidence">
                        {Math.round(alert.confidence * 100)}%
                      </span>
                      <span className="timestamp">
                        {formatTime(alert.timestamp)}
                      </span>
                    </div>
                  </div>
                </div>
              );
            })
          )}
        </div>
      </div>

      {/* Security Zones (Optional - if you want to implement zones) */}
      <div className="security-zones-section">
        <h4>Security Zones</h4>
        <div className="zones-grid">
          {securityZones.map(zone => (
            <div key={zone.id} className="zone-card">
              <div className="zone-info">
                <span className="zone-name">{zone.name}</span>
                <span className="zone-status secure">Secure</span>
              </div>
              <div className="zone-cameras">
                {/* You can implement zone-camera mapping here */}
                <span className="camera-count">0 cameras</span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Quick Actions */}
      <div className="quick-actions-section">
        <h4>Quick Actions</h4>
        <div className="action-buttons">
          <button className="action-button emergency">
            🚨 Emergency Mode
          </button>
          <button className="action-button patrol">
            🔍 Start Patrol
          </button>
          <button className="action-button silence">
            🔕 Silence Alerts (1h)
          </button>
          <button className="action-button report">
            📋 Generate Report
          </button>
        </div>
      </div>
    </div>
  );
};

export default SecurityDashboard;