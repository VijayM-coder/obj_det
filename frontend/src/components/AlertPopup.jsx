import React, { useState, useEffect } from 'react';
import './AlertPopup.css';

const AlertPopup = ({ alert, camera, onClose, onViewHistory }) => {
  const [isVisible, setIsVisible] = useState(false);
  const [timeRemaining, setTimeRemaining] = useState(15);

  useEffect(() => {
    // Show popup with animation
    setIsVisible(true);

    // Auto-close timer
    const timer = setInterval(() => {
      setTimeRemaining(prev => {
        if (prev <= 1) {
          handleClose();
          return 0;
        }
        return prev - 1;
      });
    }, 1000);

    return () => clearInterval(timer);
  }, []);

  const handleClose = () => {
    setIsVisible(false);
    setTimeout(() => {
      onClose(alert.id);
    }, 300); // Wait for animation to complete
  };

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

  const getSeverityConfig = (severity) => {
    switch (severity) {
      case 'critical':
        return {
          icon: '🚨',
          title: 'CRITICAL ALERT',
          bgColor: '#ff3b30',
          borderColor: '#ff1a1a'
        };
      case 'warning':
        return {
          icon: '⚠️',
          title: 'WARNING',
          bgColor: '#ff9500',
          borderColor: '#ff8800'
        };
      default:
        return {
          icon: '🔍',
          title: 'DETECTION',
          bgColor: '#34c759',
          borderColor: '#2d9b47'
        };
    }
  };

  const formatBoundingBox = (bbox) => {
    if (!bbox || !Array.isArray(bbox) || bbox.length !== 4) return 'N/A';
    const [x1, y1, x2, y2] = bbox.map(coord => Math.round(coord));
    const width = x2 - x1;
    const height = y2 - y1;
    return `(${x1}, ${y1}) - ${width}×${height}px`;
  };

  const timestamp = formatTimestamp(alert.timestamp);
  const cameraName = camera ? (camera.name || `Camera ${alert.cameraId}`) : `Camera ${alert.cameraId}`;
  const severity = alert.severity || 'info';
  const severityConfig = getSeverityConfig(severity);
  const confidence = Math.round(alert.confidence * 100);

  return (
    <div className={`alert-popup-overlay ${isVisible ? 'visible' : ''}`}>
      <div className={`alert-popup ${severity} ${isVisible ? 'visible' : ''}`}>
        {/* Header */}
        <div 
          className="alert-popup-header"
          style={{ backgroundColor: severityConfig.bgColor }}
        >
          <div className="header-left">
            <span className="alert-icon">{severityConfig.icon}</span>
            <div className="alert-title-group">
              <h2 className="alert-title">{severityConfig.title}</h2>
              <span className="camera-name">{cameraName}</span>
            </div>
          </div>
          
          <div className="header-right">
            <div className="auto-close-timer">
              <span className="timer-text">Auto-close in {timeRemaining}s</span>
              <div className="timer-bar">
                <div 
                  className="timer-progress"
                  style={{ width: `${(timeRemaining / 15) * 100}%` }}
                />
              </div>
            </div>
            <button 
              className="close-button"
              onClick={handleClose}
              title="Close Alert"
            >
              ✕
            </button>
          </div>
        </div>

        {/* Content */}
        <div className="alert-popup-content">
          {/* Detection Details */}
          <div className="detection-details">
            <div className="detail-row">
              <span className="detail-label">Object Detected:</span>
              <span className="detail-value object-type">{alert.objectType}</span>
            </div>
            
            <div className="detail-row">
              <span className="detail-label">Confidence:</span>
              <span 
                className="detail-value confidence"
                style={{ 
                  color: confidence >= 90 ? '#ff3b30' : 
                         confidence >= 70 ? '#ff9500' : '#34c759' 
                }}
              >
                {confidence}%
              </span>
            </div>
            
            <div className="detail-row">
              <span className="detail-label">Detection Time:</span>
              <span className="detail-value">
                {timestamp.date} at {timestamp.time}
              </span>
            </div>
            
            <div className="detail-row">
              <span className="detail-label">Position:</span>
              <span className="detail-value">{formatBoundingBox(alert.bbox)}</span>
            </div>
          </div>

          {/* Detection Image */}
          <div className="detection-image-container">
            {alert.imageData ? (
              <div className="image-wrapper">
                <img 
                  src={`data:image/jpeg;base64,${alert.imageData}`} 
                  alt={`Alert: ${alert.objectType}`}
                  className="detection-image"
                  onError={(e) => {
                    e.target.style.display = 'none';
                    e.target.nextElementSibling.style.display = 'flex';
                  }}
                />
                <div className="image-error" style={{ display: 'none' }}>
                  <span className="error-icon">🖼️</span>
                  <p>Image not available</p>
                </div>
                
                {/* Bounding Box Overlay */}
                {alert.bbox && (
                  <div className="bbox-overlay">
                    <div 
                      className="bounding-box"
                      style={{
                        left: `${(alert.bbox[0] / 640) * 100}%`,
                        top: `${(alert.bbox[1] / 480) * 100}%`,
                        width: `${((alert.bbox[2] - alert.bbox[0]) / 640) * 100}%`,
                        height: `${((alert.bbox[3] - alert.bbox[1]) / 480) * 100}%`,
                        borderColor: severityConfig.borderColor
                      }}
                    >
                      <div className="bbox-label">
                        {alert.objectType} ({confidence}%)
                      </div>
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="no-image">
                <span className="no-image-icon">🖼️</span>
                <p>No image available</p>
              </div>
            )}
          </div>
        </div>

        {/* Footer Actions */}
        <div className="alert-popup-footer">
          <div className="footer-info">
            <span className="alert-id">Alert ID: {alert.id}</span>
          </div>
          
          <div className="footer-actions">
            <button 
              className="action-button secondary"
              onClick={() => {
                onViewHistory();
                handleClose();
              }}
            >
              📋 View History
            </button>
            
            <button 
              className="action-button primary"
              onClick={handleClose}
            >
              ✓ Acknowledge
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AlertPopup;