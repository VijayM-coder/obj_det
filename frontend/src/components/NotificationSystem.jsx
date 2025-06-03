import React, { useState, useEffect } from 'react';
import './NotificationSystem.css';
import { subscribeToAlerts } from '../services/alertServices';

const NotificationSystem = ({ cameras, onViewAlert, isTrackingActive }) => {
  const [notifications, setNotifications] = useState([]);
  const [unreadCount, setUnreadCount] = useState(0);
  const [isSystemActive, setIsSystemActive] = useState(false);

  // Only activate notification system when tracking is active
  useEffect(() => {
    setIsSystemActive(isTrackingActive);
    
    if (!isTrackingActive) {
      // Clear notifications when tracking stops
      setNotifications([]);
      setUnreadCount(0);
    }
  }, [isTrackingActive]);

  useEffect(() => {
    if (!isSystemActive) {
      return; // Don't subscribe if system is not active
    }

    console.log('NotificationSystem: Setting up alert subscriptions');

    // Subscribe to all camera alerts when system is active
    const unsubscribe = subscribeToAlerts('all', (newAlert) => {
      console.log('NotificationSystem: Received new alert:', newAlert);

      // Create notification object
      const notification = {
        ...newAlert,
        id: newAlert.id || `${newAlert.cameraId}_${Date.now()}_${Math.random()}`,
        receivedAt: Date.now()
      };

      // Add the new notification (keep only 5 most recent)
      setNotifications(prev => {
        const updated = [notification, ...prev.slice(0, 4)];
        return updated;
      });
      
      // Increment unread count
      setUnreadCount(prev => prev + 1);

      // Play system notification sound if available
      if (window.electron) {
        window.electron.playBeep().catch(console.error);
      }

      // Auto-dismiss notification after 8 seconds
      setTimeout(() => {
        setNotifications(prev => 
          prev.filter(alert => alert.id !== notification.id)
        );
      }, 8000);
    });

    return () => {
      console.log('NotificationSystem: Cleaning up alert subscriptions');
      unsubscribe();
    };
  }, [isSystemActive]);

  // Get camera name by ID
  const getCameraName = (cameraId) => {
    const camera = cameras.find(cam => cam.id === cameraId);
    return camera ? (camera.name || `Camera ${cameraId}`) : `Camera ${cameraId}`;
  };

  // Handle clicking on a notification
  const handleNotificationClick = (notification) => {
    console.log('NotificationSystem: Notification clicked:', notification);
    
    // Pass the alert to parent for handling
    if (onViewAlert) {
      onViewAlert(notification);
    }
    
    // Remove this notification
    setNotifications(prev => 
      prev.filter(alert => alert.id !== notification.id)
    );

    // Decrease unread count
    setUnreadCount(prev => Math.max(0, prev - 1));
  };

  // Mark all as read
  const clearAllNotifications = () => {
    setNotifications([]);
    setUnreadCount(0);
  };

  // Format timestamp for display
  const formatTimestamp = (isoString) => {
    try {
      const date = new Date(isoString);
      return date.toLocaleTimeString();
    } catch (e) {
      return new Date().toLocaleTimeString();
    }
  };

  // Get confidence color
  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.9) return '#ff4444'; // High confidence - red
    if (confidence >= 0.7) return '#ff8800'; // Medium confidence - orange
    return '#44ff44'; // Lower confidence - green
  };

  // Don't render anything if tracking is not active
  if (!isSystemActive) {
    return null;
  }

  return (
    <div className="notification-system">
      {/* Active system indicator */}
      <div className="notification-status">
        <div className="status-indicator active">
          <div className="pulse-dot"></div>
          <span>Alert System Active</span>
        </div>
      </div>

      {/* Notifications container */}
      {notifications.length > 0 && (
        <div className="notification-container">
          <div className="notification-header">
            <span className="notification-title">Detection Alerts</span>
            {notifications.length > 1 && (
              <button 
                className="clear-all-button"
                onClick={clearAllNotifications}
                title="Clear all notifications"
              >
                Clear All ({notifications.length})
              </button>
            )}
          </div>

          <div className="notification-list">
            {notifications.map(notification => (
              <div 
                key={notification.id} 
                className="notification-item"
                onClick={() => handleNotificationClick(notification)}
              >
                <div className="notification-content">
                  <div className="notification-main">
                    <div className="notification-info">
                      <span className="notification-object">
                        {notification.objectType}
                      </span>
                      <span className="notification-camera">
                        {getCameraName(notification.cameraId)}
                      </span>
                    </div>
                    
                    <div className="notification-meta">
                      <span className="notification-time">
                        {formatTimestamp(notification.timestamp)}
                      </span>
                      <span 
                        className="notification-confidence"
                        style={{ color: getConfidenceColor(notification.confidence) }}
                      >
                        {Math.round(notification.confidence * 100)}%
                      </span>
                    </div>
                  </div>
                  
                  {notification.imageData && (
                    <div className="notification-thumbnail">
                      <img 
                        src={`data:image/jpeg;base64,${notification.imageData}`} 
                        alt={`Alert: ${notification.objectType}`}
                        onError={(e) => {
                          e.target.style.display = 'none';
                        }}
                      />
                    </div>
                  )}
                </div>
                
                <div className="notification-actions">
                  <button 
                    className="view-button"
                    onClick={(e) => {
                      e.stopPropagation();
                      handleNotificationClick(notification);
                    }}
                  >
                    View
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
      
      {/* Unread count badge */}
      {unreadCount > 0 && notifications.length === 0 && (
        <div className="notification-badge" onClick={() => onViewAlert()}>
          <span className="badge-count">{unreadCount}</span>
          <span className="badge-text">New Alerts</span>
        </div>
      )}
    </div>
  );
};

export default NotificationSystem;