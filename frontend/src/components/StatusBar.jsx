// components/StatusBar.jsx - Bottom status bar component
import React, { useState, useEffect } from 'react';
import './StatusBar.css';

const StatusBar = ({ 
  cameras, 
  trackingPairs, 
  isTracking, 
  alertCount, 
  criticalAlertCount,
  activeConnections,
  systemUptime 
}) => {
  const [uptime, setUptime] = useState('00:00:00');
  const [currentTime, setCurrentTime] = useState(new Date());

  // Update uptime display
  useEffect(() => {
    if (!isTracking || !systemUptime) {
      setUptime('00:00:00');
      return;
    }

    const interval = setInterval(() => {
      const elapsed = Date.now() - systemUptime;
      const hours = Math.floor(elapsed / (1000 * 60 * 60));
      const minutes = Math.floor((elapsed % (1000 * 60 * 60)) / (1000 * 60));
      const seconds = Math.floor((elapsed % (1000 * 60)) / 1000);
      
      setUptime(
        `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`
      );
    }, 1000);

    return () => clearInterval(interval);
  }, [isTracking, systemUptime]);

  // Update current time
  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentTime(new Date());
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  // Get system status
  const getSystemStatus = () => {
    if (!isTracking) return { status: 'offline', color: '#999' };
    
    if (activeConnections === trackingPairs.length && trackingPairs.length > 0) {
      return { status: 'optimal', color: '#34c759' };
    } else if (activeConnections > 0) {
      return { status: 'partial', color: '#ff9500' };
    } else {
      return { status: 'connecting', color: '#007aff' };
    }
  };

  const systemStatus = getSystemStatus();

  return (
    <footer className="status-bar">
      <div className="status-section">
        <div className="status-item">
          <span className="status-label">System:</span>
          <div className="status-indicator">
            <span 
              className="status-dot"
              style={{ backgroundColor: systemStatus.color }}
            ></span>
            <span className="status-text">{systemStatus.status}</span>
          </div>
        </div>

        <div className="status-item">
          <span className="status-label">Cameras:</span>
          <span className="status-value">{cameras.length}</span>
        </div>

        <div className="status-item">
          <span className="status-label">Tracking:</span>
          <span className="status-value">{trackingPairs.length}</span>
        </div>

        <div className="status-item">
          <span className="status-label">Active:</span>
          <span className="status-value">{activeConnections}</span>
        </div>
      </div>

      <div className="status-section">
        <div className="status-item">
          <span className="status-label">Alerts:</span>
          <span className={`status-value ${alertCount > 0 ? 'has-alerts' : ''}`}>
            {alertCount}
          </span>
        </div>

        {criticalAlertCount > 0 && (
          <div className="status-item critical">
            <span className="status-label">Critical:</span>
            <span className="status-value critical">{criticalAlertCount}</span>
          </div>
        )}
      </div>

      <div className="status-section">
        <div className="status-item">
          <span className="status-label">Uptime:</span>
          <span className="status-value uptime">{uptime}</span>
        </div>

        <div className="status-item">
          <span className="status-label">Time:</span>
          <span className="status-value time">
            {currentTime.toLocaleTimeString()}
          </span>
        </div>
      </div>

      <div className="status-section">
        <div className="system-info">
          <span className="app-version">v2.1.0</span>
          <span className="separator">|</span>
          <span className="system-name">AI Security Monitor</span>
        </div>
      </div>
    </footer>
  );
};

export default StatusBar;