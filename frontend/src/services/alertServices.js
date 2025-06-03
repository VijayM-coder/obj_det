// services/alertServices.js
import { EventEmitter } from 'events';

class AlertManager extends EventEmitter {
  constructor() {
    super();
    this.alertConnections = new Map(); // Map of camera_id -> WebSocket
    this.alertSubscribers = new Map(); // Map of camera_id -> Set of callbacks
    this.alertHistory = new Map(); // Map of camera_id -> Array of alerts
    this.connectionStatus = new Map(); // Map of camera_id -> status
    this.reconnectTimeouts = new Map();
    this.maxRetries = 5;
    this.baseRetryDelay = 1000; // 1 second
  }

  // Connect to alert WebSocket for a specific camera
  connectToAlertWebSocket(cameraId) {
    const stringCameraId = String(cameraId);
    
    // Don't create duplicate connections
    if (this.alertConnections.has(stringCameraId)) {
      const existingWs = this.alertConnections.get(stringCameraId);
      if (existingWs.readyState === WebSocket.OPEN) {
        console.log(`Alert WebSocket already connected for camera ${cameraId}`);
        return this.getConnectionAPI(stringCameraId);
      } else {
        // Close existing connection if it's not open
        existingWs.close();
        this.alertConnections.delete(stringCameraId);
      }
    }

    console.log(`Connecting alert WebSocket for camera ${cameraId}`);
    
    try {
      const ws = new WebSocket(`ws://localhost:8001/ws/cameras/${cameraId}`);
      
      ws.onopen = () => {
        console.log(`Alert WebSocket connected for camera ${cameraId}`);
        this.alertConnections.set(stringCameraId, ws);
        this.connectionStatus.set(stringCameraId, 'connected');
        this.emit('websocket-status', { cameraId: stringCameraId, status: 'connected' });
        
        // Clear any reconnection timeout
        if (this.reconnectTimeouts.has(stringCameraId)) {
          clearTimeout(this.reconnectTimeouts.get(stringCameraId));
          this.reconnectTimeouts.delete(stringCameraId);
        }

        // Send a ping to keep connection alive
        this.startHeartbeat(stringCameraId);
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          
          if (data.type === 'alert') {
            const alert = {
              id: `${data.camera_id}_${Date.now()}_${Math.random()}`,
              cameraId: data.camera_id,
              objectType: data.alert.object_type,
              confidence: data.alert.confidence,
              bbox: data.alert.bbox,
              timestamp: data.alert.timestamp,
              imageData: data.alert.image_data
            };

            // Store in history
            this.addToHistory(stringCameraId, alert);
            
            // Notify subscribers
            this.notifySubscribers(stringCameraId, alert);
            
            // Emit global alert event
            this.emit('alert', alert);
            
            console.log(`Alert received for camera ${cameraId}:`, alert.objectType);
          }
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };

      ws.onerror = (error) => {
        console.error(`Alert WebSocket error for camera ${cameraId}:`, error);
        this.connectionStatus.set(stringCameraId, 'error');
        this.emit('websocket-status', { cameraId: stringCameraId, status: 'error' });
      };

      ws.onclose = (event) => {
        console.log(`Alert WebSocket closed for camera ${cameraId}`, event.code, event.reason);
        this.alertConnections.delete(stringCameraId);
        this.connectionStatus.set(stringCameraId, 'closed');
        this.emit('websocket-status', { cameraId: stringCameraId, status: 'closed' });
        
        // Stop heartbeat
        this.stopHeartbeat(stringCameraId);
        
        // Attempt to reconnect if it wasn't a manual close
        if (event.code !== 1000) {
          this.scheduleReconnect(stringCameraId);
        }
      };

    } catch (error) {
      console.error(`Failed to create WebSocket for camera ${cameraId}:`, error);
      this.connectionStatus.set(stringCameraId, 'error');
      this.emit('websocket-status', { cameraId: stringCameraId, status: 'error' });
    }

    return this.getConnectionAPI(stringCameraId);
  }

  // Disconnect alert WebSocket for a specific camera
  disconnectAlertWebSocket(cameraId) {
    const stringCameraId = String(cameraId);
    
    if (this.alertConnections.has(stringCameraId)) {
      const ws = this.alertConnections.get(stringCameraId);
      this.stopHeartbeat(stringCameraId);
      ws.close(1000, 'Manual disconnect'); // Normal closure
      this.alertConnections.delete(stringCameraId);
      console.log(`Alert WebSocket disconnected for camera ${cameraId}`);
    }

    // Clear reconnection timeout
    if (this.reconnectTimeouts.has(stringCameraId)) {
      clearTimeout(this.reconnectTimeouts.get(stringCameraId));
      this.reconnectTimeouts.delete(stringCameraId);
    }

    this.connectionStatus.set(stringCameraId, 'disconnected');
    this.emit('websocket-status', { cameraId: stringCameraId, status: 'disconnected' });
  }

  // Get connection API for external use
  getConnectionAPI(cameraId) {
    const stringCameraId = String(cameraId);
    return {
      disconnect: () => this.disconnectAlertWebSocket(stringCameraId),
      reconnect: () => this.connectToAlertWebSocket(stringCameraId),
      isConnected: () => {
        const ws = this.alertConnections.get(stringCameraId);
        return ws && ws.readyState === WebSocket.OPEN;
      },
      getStatus: () => this.connectionStatus.get(stringCameraId) || 'disconnected'
    };
  }

  // Schedule reconnection with exponential backoff
  scheduleReconnect(cameraId, retryCount = 0) {
    const stringCameraId = String(cameraId);
    
    if (retryCount >= this.maxRetries) {
      console.warn(`Max reconnection attempts reached for camera ${cameraId}`);
      this.connectionStatus.set(stringCameraId, 'failed');
      this.emit('websocket-status', { cameraId: stringCameraId, status: 'failed' });
      return;
    }

    const delay = this.baseRetryDelay * Math.pow(2, retryCount);
    console.log(`Scheduling reconnection for camera ${cameraId} in ${delay}ms (attempt ${retryCount + 1})`);

    const timeout = setTimeout(() => {
      console.log(`Attempting to reconnect camera ${cameraId} (attempt ${retryCount + 1})`);
      this.connectToAlertWebSocket(cameraId);
    }, delay);

    this.reconnectTimeouts.set(stringCameraId, timeout);
  }

  // Start heartbeat to keep connection alive
  startHeartbeat(cameraId) {
    const stringCameraId = String(cameraId);
    const ws = this.alertConnections.get(stringCameraId);
    
    if (!ws) return;

    const heartbeatInterval = setInterval(() => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'ping', timestamp: Date.now() }));
      } else {
        clearInterval(heartbeatInterval);
      }
    }, 30000); // 30 seconds

    // Store interval reference for cleanup
    ws._heartbeatInterval = heartbeatInterval;
  }

  // Stop heartbeat
  stopHeartbeat(cameraId) {
    const stringCameraId = String(cameraId);
    const ws = this.alertConnections.get(stringCameraId);
    
    if (ws && ws._heartbeatInterval) {
      clearInterval(ws._heartbeatInterval);
      delete ws._heartbeatInterval;
    }
  }

  // Add alert to history
  addToHistory(cameraId, alert) {
    const stringCameraId = String(cameraId);
    
    if (!this.alertHistory.has(stringCameraId)) {
      this.alertHistory.set(stringCameraId, []);
    }

    const history = this.alertHistory.get(stringCameraId);
    history.unshift(alert); // Add to beginning

    // Keep only last 100 alerts per camera
    if (history.length > 100) {
      history.splice(100);
    }

    // Save to persistent storage
    this.saveAlertsToStorage();
  }

  // Notify subscribers
  notifySubscribers(cameraId, alert) {
    const stringCameraId = String(cameraId);
    
    // Notify specific camera subscribers
    if (this.alertSubscribers.has(stringCameraId)) {
      this.alertSubscribers.get(stringCameraId).forEach(callback => {
        try {
          callback(alert);
        } catch (error) {
          console.error('Error in alert subscriber callback:', error);
        }
      });
    }

    // Notify global subscribers (subscribed to 'all')
    if (this.alertSubscribers.has('all')) {
      this.alertSubscribers.get('all').forEach(callback => {
        try {
          callback(alert);
        } catch (error) {
          console.error('Error in global alert subscriber callback:', error);
        }
      });
    }
  }

  // Subscribe to alerts for a specific camera or all cameras
  subscribeToAlerts(cameraId, callback) {
    const stringCameraId = String(cameraId);
    
    if (!this.alertSubscribers.has(stringCameraId)) {
      this.alertSubscribers.set(stringCameraId, new Set());
    }

    this.alertSubscribers.get(stringCameraId).add(callback);

    // Return unsubscribe function
    return () => {
      if (this.alertSubscribers.has(stringCameraId)) {
        this.alertSubscribers.get(stringCameraId).delete(callback);
        
        // Clean up empty sets
        if (this.alertSubscribers.get(stringCameraId).size === 0) {
          this.alertSubscribers.delete(stringCameraId);
        }
      }
    };
  }

  // Subscribe to WebSocket status changes
  subscribeToWebSocketStatus(callback) {
    this.on('websocket-status', callback);
    return () => this.off('websocket-status', callback);
  }

  // Get recent alerts for a camera
  getRecentAlerts(cameraId, limit = 50) {
    const stringCameraId = String(cameraId);
    const history = this.alertHistory.get(stringCameraId) || [];
    return history.slice(0, limit);
  }

  // Clear alerts for a camera
  clearAlerts(cameraId) {
    const stringCameraId = String(cameraId);
    this.alertHistory.set(stringCameraId, []);
    this.saveAlertsToStorage();
  }

  // Save alerts to persistent storage
  async saveAlertsToStorage() {
    if (window.electron) {
      try {
        const alertsData = {};
        for (const [cameraId, alerts] of this.alertHistory.entries()) {
          alertsData[cameraId] = alerts;
        }
        await window.electron.saveAlertData({ alerts: alertsData });
      } catch (error) {
        console.error('Failed to save alerts to storage:', error);
      }
    }
  }

  // Load alerts from persistent storage
  async loadAlertsFromStorage() {
    if (window.electron) {
      try {
        const data = await window.electron.loadAlertData();
        if (data.alerts) {
          for (const [cameraId, alerts] of Object.entries(data.alerts)) {
            this.alertHistory.set(cameraId, alerts);
          }
        }
      } catch (error) {
        console.error('Failed to load alerts from storage:', error);
      }
    }
  }

  // Clear all alert data
  async clearAllAlertData() {
    this.alertHistory.clear();
    if (window.electron) {
      try {
        await window.electron.clearAlertData();
      } catch (error) {
        console.error('Failed to clear alert data:', error);
      }
    }
  }

  // Get connection status for a camera
  getConnectionStatus(cameraId) {
    const stringCameraId = String(cameraId);
    return this.connectionStatus.get(stringCameraId) || 'disconnected';
  }

  // Check if tracking is active for any camera
  isTrackingActive() {
    return this.alertConnections.size > 0;
  }

  // Cleanup all connections
  cleanup() {
    console.log('Cleaning up alert connections...');
    
    // Clear all timeouts
    for (const timeout of this.reconnectTimeouts.values()) {
      clearTimeout(timeout);
    }
    this.reconnectTimeouts.clear();

    // Close all WebSocket connections
    for (const [cameraId, ws] of this.alertConnections.entries()) {
      this.stopHeartbeat(cameraId);
      if (ws.readyState === WebSocket.OPEN) {
        ws.close(1000, 'Application shutdown');
      }
    }
    this.alertConnections.clear();

    // Clear all subscribers
    this.alertSubscribers.clear();
    this.removeAllListeners();
  }
}

// Create singleton instance
const alertManager = new AlertManager();

// Load alerts from storage on initialization
alertManager.loadAlertsFromStorage();

// Export functions for backward compatibility
export const connectToAlertWebSocket = (cameraId) => {
  return alertManager.connectToAlertWebSocket(cameraId);
};

export const disconnectAlertWebSocket = (cameraId) => {
  return alertManager.disconnectAlertWebSocket(cameraId);
};

export const subscribeToAlerts = (cameraId, callback) => {
  return alertManager.subscribeToAlerts(cameraId, callback);
};

export const subscribeToWebSocketStatus = (callback) => {
  return alertManager.subscribeToWebSocketStatus(callback);
};

export const getRecentAlerts = (cameraId, limit) => {
  return alertManager.getRecentAlerts(cameraId, limit);
};

export const clearAlerts = (cameraId) => {
  return alertManager.clearAlerts(cameraId);
};

export const getConnectionStatus = (cameraId) => {
  return alertManager.getConnectionStatus(cameraId);
};

export const isTrackingActive = () => {
  return alertManager.isTrackingActive();
};

// Export the manager for advanced usage
export { alertManager };

// Cleanup on page unload
if (typeof window !== 'undefined') {
  window.addEventListener('beforeunload', () => {
    alertManager.cleanup();
  });
}