// components/AlertToast.jsx - Enhanced toast notifications for alerts
import React, { useState, useEffect } from 'react';
import './AlertToast.css';

const AlertToast = ({ alert, cameras, onView, onDismiss, onAcknowledge }) => {
  const [isVisible, setIsVisible] = useState(false);
  const [timeLeft, setTimeLeft] = useState(5);

  useEffect(() => {
    // Animate in
    setIsVisible(true);
    
    // Countdown timer
    const timer = setInterval(() => {
      setTimeLeft(prev => {
        if (prev <= 1) {
          onDismiss();
          return 0;
        }
        return prev - 1;
      });
    }, 1000);
    
    return () => clearInterval(timer);
  }, [onDismiss]);

  // Get camera name
  const getCameraName = (cameraId) => {
    const camera = cameras.find(cam => cam.id === cameraId);
    return camera ? (camera.name || `Camera ${cameraId}`) : `Camera ${cameraId}`;
  };

  // Get severity config
  const getSeverityConfig = (severity) => {
    switch (severity) {
      case 'critical':
        return {
          icon: '🔴',
          label: 'CRITICAL ALERT',
          color: '#ff3b30',
          bgColor: '#ffebee'
        };
      case 'warning':
        return {
          icon: '🟡',
          label: 'WARNING',
          color: '#ff9500',
          bgColor: '#fff8e1'
        };
      default:
        return {
          icon: '🟢',
          label: 'DETECTION',
          color: '#34c759',
          bgColor: '#e8f5e8'
        };
    }
  };

  // Format timestamp
  const formatTime = (timestamp) => {
    try {
      return new Date(timestamp).toLocaleTimeString();
    } catch {
      return new Date().toLocaleTimeString();
    }
  };

  const severityConfig = getSeverityConfig(alert.severity);
  const confidence = Math.round(alert.confidence * 100);

  return (
    <div 
      className={`alert-toast ${alert.severity} ${isVisible ? 'visible' : ''}`}
      style={{ 
        borderLeft: `4px solid ${severityConfig.color}`,
        backgroundColor: severityConfig.bgColor
      }}
    >
      {/* Toast Header */}
      <div className="toast-header">
        <div className="severity-info">
          <span className="severity-icon">{severityConfig.icon}</span>
          <span className="severity-label" style={{ color: severityConfig.color }}>
            {severityConfig.label}
          </span>
        </div>
        <div className="toast-actions">
          <span className="time-left">{timeLeft}s</span>
          <button 
            className="close-button"
            onClick={onDismiss}
            title="Dismiss"
          >
            ✕
          </button>
        </div>
      </div>

      {/* Alert Content */}
      <div className="toast-content">
        <div className="alert-main-info">
          <div className="object-detection">
            <span className="object-type">{alert.objectType}</span>
            <span className="confidence-badge" style={{ backgroundColor: severityConfig.color }}>
              {confidence}%
            </span>
          </div>
          
          <div className="location-info">
            <span className="camera-name">{getCameraName(alert.cameraId)}</span>
            <span className="detection-time">{formatTime(alert.timestamp)}</span>
          </div>
        </div>

        {/* Alert Image */}
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
      </div>

      {/* Action Buttons */}
      <div className="toast-buttons">
        <button 
          className="view-button"
          onClick={() => {
            onView(alert);
            onDismiss();
          }}
        >
          🕵 View Camera
        </button>
        
        <button 
          className="acknowledge-button"
          onClick={() => {
            onAcknowledge(alert.id);
          }}
        >
          ✅ Acknowledge
        </button>
        
        {alert.severity === 'critical' && (
          <button 
            className="priority-button"
            onClick={() => {
              // Could trigger additional actions like sending notifications
              onView(alert);
              onDismiss();
            }}
          >
            🚨 Priority View
          </button>
        )}
      </div>

      {/* Progress bar */}
      <div 
        className="progress-bar"
        style={{
          width: `${(timeLeft / 5) * 100}%`,
          backgroundColor: severityConfig.color
        }}
      />
    </div>
  );
};

export default AlertToast;