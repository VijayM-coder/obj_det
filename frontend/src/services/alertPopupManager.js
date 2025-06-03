// services/alertPopupManager.js
import { EventEmitter } from 'events';

class AlertPopupManager extends EventEmitter {
  constructor() {
    super();
    this.activePopups = new Map(); // alertId -> popup data
    this.popupQueue = [];
    this.maxActivePopups = 3;
    this.settings = {
      autoCloseDelay: 15000, // 15 seconds
      enableSound: true,
      enablePopups: true,
      severityFilters: {
        critical: true,
        warning: true,
        info: true
      }
    };
  }

  // Show popup for alert
  showPopup(alert, camera) {
    if (!this.settings.enablePopups) {
      console.log('Popups disabled, skipping alert popup');
      return;
    }

    // Check if popup should be shown based on severity filter
    const severity = alert.severity || 'info';
    if (!this.settings.severityFilters[severity]) {
      console.log(`Popup filtered out for severity: ${severity}`);
      return;
    }

    // Enhance alert data
    const enhancedAlert = {
      ...alert,
      id: alert.id || `${alert.cameraId}_${Date.now()}_${Math.random()}`,
      severity: severity,
      timestamp: alert.timestamp || new Date().toISOString(),
      showTime: Date.now()
    };

    // If we have too many active popups, queue this one
    if (this.activePopups.size >= this.maxActivePopups) {
      this.popupQueue.push({ alert: enhancedAlert, camera });
      console.log('Popup queued, too many active popups');
      return;
    }

    // Show the popup
    this.displayPopup(enhancedAlert, camera);
  }

  // Display popup
  displayPopup(alert, camera) {
    console.log('Displaying popup for alert:', alert.id);

    // Store popup data
    this.activePopups.set(alert.id, {
      alert,
      camera,
      startTime: Date.now()
    });

    // Emit event to show popup in UI
    this.emit('show-popup', { alert, camera });

    // Set up auto-close timer
    setTimeout(() => {
      this.closePopup(alert.id, 'auto-close');
    }, this.settings.autoCloseDelay);

    // Play sound if enabled
    if (this.settings.enableSound) {
      this.playAlertSound(alert.severity);
    }

    // Track popup analytics
    this.trackPopupEvent('shown', alert);
  }

  // Close popup
  closePopup(alertId, reason = 'manual') {
    if (!this.activePopups.has(alertId)) {
      return;
    }

    const popupData = this.activePopups.get(alertId);
    console.log(`Closing popup ${alertId}, reason: ${reason}`);

    // Remove from active popups
    this.activePopups.delete(alertId);

    // Emit close event
    this.emit('close-popup', { alertId, reason });

    // Track analytics
    this.trackPopupEvent('closed', popupData.alert, { 
      reason, 
      duration: Date.now() - popupData.startTime 
    });

    // Show next popup from queue if available
    if (this.popupQueue.length > 0) {
      const nextPopup = this.popupQueue.shift();
      setTimeout(() => {
        this.displayPopup(nextPopup.alert, nextPopup.camera);
      }, 500); // Small delay to avoid overwhelming
    }
  }

  // Close all popups
  closeAllPopups(reason = 'manual') {
    const alertIds = Array.from(this.activePopups.keys());
    alertIds.forEach(alertId => {
      this.closePopup(alertId, reason);
    });
    
    // Clear queue as well
    this.popupQueue = [];
  }

  // Acknowledge alert (closes popup and marks as acknowledged)
  acknowledgeAlert(alertId) {
    this.closePopup(alertId, 'acknowledged');
    this.emit('alert-acknowledged', { alertId });
  }

  // Play alert sound based on severity
  playAlertSound(severity) {
    try {
      // Try to use Electron's native beep first
      if (window.electron && window.electron.playBeep) {
        window.electron.playBeep();
        return;
      }

      // Fallback to Web Audio API
      const context = new (window.AudioContext || window.webkitAudioContext)();
      const oscillator = context.createOscillator();
      const gainNode = context.createGain();
      
      oscillator.connect(gainNode);
      gainNode.connect(context.destination);
      
      // Set frequency and volume based on severity
      switch (severity) {
        case 'critical':
          oscillator.frequency.value = 800;
          gainNode.gain.value = 0.8;
          // Play multiple beeps for critical
          this.playBeepSequence(context, [800, 1000, 800], [0.2, 0.2, 0.2]);
          break;
        case 'warning':
          oscillator.frequency.value = 600;
          gainNode.gain.value = 0.6;
          oscillator.start();
          oscillator.stop(context.currentTime + 0.3);
          break;
        default:
          oscillator.frequency.value = 400;
          gainNode.gain.value = 0.4;
          oscillator.start();
          oscillator.stop(context.currentTime + 0.2);
      }
    } catch (error) {
      console.error('Error playing alert sound:', error);
    }
  }

  // Play sequence of beeps
  playBeepSequence(context, frequencies, durations) {
    let currentTime = context.currentTime;
    
    frequencies.forEach((freq, index) => {
      const osc = context.createOscillator();
      const gain = context.createGain();
      
      osc.connect(gain);
      gain.connect(context.destination);
      
      osc.frequency.value = freq;
      gain.gain.value = 0.6;
      
      osc.start(currentTime);
      osc.stop(currentTime + durations[index]);
      
      currentTime += durations[index] + 0.1; // Small gap between beeps
    });
  }

  // Update settings
  updateSettings(newSettings) {
    this.settings = { ...this.settings, ...newSettings };
    console.log('Popup manager settings updated:', this.settings);
    this.emit('settings-updated', this.settings);
  }

  // Get current settings
  getSettings() {
    return { ...this.settings };
  }

  // Get popup statistics
  getStatistics() {
    return {
      activePopups: this.activePopups.size,
      queuedPopups: this.popupQueue.length,
      totalShown: this.analytics?.totalShown || 0,
      totalClosed: this.analytics?.totalClosed || 0,
      averageDuration: this.analytics?.averageDuration || 0
    };
  }

  // Track popup events for analytics
  trackPopupEvent(event, alert, metadata = {}) {
    if (!this.analytics) {
      this.analytics = {
        totalShown: 0,
        totalClosed: 0,
        totalDuration: 0,
        severityBreakdown: { critical: 0, warning: 0, info: 0 },
        closeReasons: {}
      };
    }

    switch (event) {
      case 'shown':
        this.analytics.totalShown++;
        this.analytics.severityBreakdown[alert.severity]++;
        break;
      case 'closed':
        this.analytics.totalClosed++;
        if (metadata.duration) {
          this.analytics.totalDuration += metadata.duration;
        }
        if (metadata.reason) {
          this.analytics.closeReasons[metadata.reason] = 
            (this.analytics.closeReasons[metadata.reason] || 0) + 1;
        }
        break;
    }

    // Calculate average duration
    if (this.analytics.totalClosed > 0) {
      this.analytics.averageDuration = this.analytics.totalDuration / this.analytics.totalClosed;
    }
  }

  // Check if alert should show popup (can be overridden with custom logic)
  shouldShowPopup(alert, camera) {
    // Don't show popup for same camera if one is already active
    const existingPopup = Array.from(this.activePopups.values())
      .find(popup => popup.alert.cameraId === alert.cameraId);
    
    if (existingPopup) {
      // Only show if this is higher severity
      const severityLevels = { info: 1, warning: 2, critical: 3 };
      const currentLevel = severityLevels[existingPopup.alert.severity] || 1;
      const newLevel = severityLevels[alert.severity] || 1;
      
      if (newLevel <= currentLevel) {
        return false;
      }
      
      // Close existing popup to show higher severity one
      this.closePopup(existingPopup.alert.id, 'replaced');
    }

    return true;
  }

  // Clean up resources
  cleanup() {
    console.log('Cleaning up popup manager...');
    this.closeAllPopups('cleanup');
    this.removeAllListeners();
  }
}

// Create singleton instance
const alertPopupManager = new AlertPopupManager();

// Export functions for easy usage
export const showAlertPopup = (alert, camera) => {
  return alertPopupManager.showPopup(alert, camera);
};

export const closeAlertPopup = (alertId, reason) => {
  return alertPopupManager.closePopup(alertId, reason);
};

export const acknowledgeAlert = (alertId) => {
  return alertPopupManager.acknowledgeAlert(alertId);
};

export const updatePopupSettings = (settings) => {
  return alertPopupManager.updateSettings(settings);
};

export const getPopupSettings = () => {
  return alertPopupManager.getSettings();
};

export const getPopupStatistics = () => {
  return alertPopupManager.getStatistics();
};

export const subscribeToPopupEvents = (callback) => {
  alertPopupManager.on('show-popup', callback);
  alertPopupManager.on('close-popup', callback);
  alertPopupManager.on('alert-acknowledged', callback);
  
  return () => {
    alertPopupManager.off('show-popup', callback);
    alertPopupManager.off('close-popup', callback);
    alertPopupManager.off('alert-acknowledged', callback);
  };
};

// Export the manager for advanced usage
export { alertPopupManager };

// Cleanup on page unload
if (typeof window !== 'undefined') {
  window.addEventListener('beforeunload', () => {
    alertPopupManager.cleanup();
  });
}