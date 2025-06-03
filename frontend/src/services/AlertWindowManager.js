// src/services/AlertWindowManager.js
// This service manages alert windows and sounds for object detection

// Store active alert windows by camera ID
const activeAlertWindows = {};

// Generate HTML content for alert window
const generateAlertHTML = (alert, camera) => {
  const cameraName = camera ? (camera.name || `Camera ${alert.cameraId}`) : `Camera ${alert.cameraId}`;
  const timestamp = new Date(alert.timestamp).toLocaleString();
  const confidence = Math.round(alert.confidence * 100);
  
  let imageHtml = '';
  if (alert.imageData) {
    imageHtml = `<img src="data:image/jpeg;base64,${alert.imageData}" alt="${alert.objectType}" style="max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">`;
  } else {
    imageHtml = `<div style="width: 100%; height: 200px; background-color: #f5f5f5; display: flex; justify-content: center; align-items: center; border-radius: 8px;">No image available</div>`;
  }
  
  return `
  <!DOCTYPE html>
  <html>
  <head>
    <title>Alert: ${alert.objectType} Detected</title>
    <style>
      body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        margin: 0;
        padding: 0;
        background-color: #f9f9f9;
        color: #333;
        overflow: hidden;
      }
      
      .alert-container {
        max-width: 100%;
        padding: 20px;
        box-sizing: border-box;
        animation: slideIn 0.3s ease-out;
      }
      
      @keyframes slideIn {
        from {
          opacity: 0;
          transform: translateY(-20px);
        }
        to {
          opacity: 1;
          transform: translateY(0);
        }
      }
      
      .alert-header {
        background-color: #ff3b30;
        color: white;
        padding: 15px 20px;
        border-radius: 8px 8px 0 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
      }
      
      .alert-title {
        margin: 0;
        font-size: 20px;
        font-weight: bold;
      }
      
      .camera-name {
        font-size: 14px;
        background-color: rgba(255, 255, 255, 0.2);
        padding: 5px 10px;
        border-radius: 15px;
      }
      
      .alert-body {
        background-color: white;
        border-radius: 0 0 8px 8px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
      }
      
      .alert-details {
        display: flex;
        justify-content: space-between;
        margin-bottom: 15px;
        font-size: 15px;
      }
      
      .alert-time {
        color: #666;
      }
      
      .alert-confidence {
        font-weight: bold;
        color: #34c759;
      }
      
      .alert-image {
        width: 100%;
        margin-top: 10px;
        margin-bottom: 20px;
        text-align: center;
      }
      
      .button-container {
        display: flex;
        justify-content: center;
        margin-top: 20px;
      }
      
      .close-button {
        background-color: #f2f2f2;
        border: none;
        padding: 10px 20px;
        border-radius: 8px;
        font-size: 14px;
        font-weight: 500;
        cursor: pointer;
        transition: background-color 0.2s;
      }
      
      .close-button:hover {
        background-color: #e0e0e0;
      }
      
      .alert-footer {
        text-align: center;
        margin-top: 15px;
        color: #999;
        font-size: 12px;
      }
    </style>
  </head>
  <body>
    <div class="alert-container">
      <div class="alert-header">
        <h1 class="alert-title">${alert.objectType} Detected</h1>
        <span class="camera-name">${cameraName}</span>
      </div>
      
      <div class="alert-body">
        <div class="alert-details">
          <span class="alert-time">Time: ${timestamp}</span>
          <span class="alert-confidence">Confidence: ${confidence}%</span>
        </div>
        
        <div class="alert-image">
          ${imageHtml}
        </div>
        
        <div class="button-container">
          <button class="close-button" onclick="window.close()">Close</button>
        </div>
        
        <div class="alert-footer">
          This window will automatically close in 20 seconds
        </div>
      </div>
    </div>
    
    <script>
      // Auto-close after 20 seconds
      setTimeout(() => window.close(), 20000);
      
      // Flash window to get attention (Windows only)
      if (window.navigator.userAgent.indexOf('Windows') !== -1) {
        const flashCount = 5;
        let currentFlash = 0;
        
        const flashInterval = setInterval(() => {
          if (currentFlash >= flashCount * 2) {
            clearInterval(flashInterval);
            return;
          }
          
          try {
            // Use requestAnimationFrame to make flashing smoother
            window.requestAnimationFrame(() => {
              document.title = currentFlash % 2 === 0 ? 
                '⚠️ ALERT: ${alert.objectType} Detected' : 
                'Alert: ${alert.objectType} Detected';
            });
            
            currentFlash++;
          } catch (e) {
            clearInterval(flashInterval);
          }
        }, 500);
      }
    </script>
  </body>
  </html>
  `;
};

// Open a new alert window
export const openAlertWindow = (alert, camera) => {
  try {
    // Make sure we have electron available
    if (!window.electron || !window.electron.openAlertWindow) {
      console.error('Electron API not available for alert windows');
      return null;
    }
    
    // Check if we already have an alert window for this camera
    if (activeAlertWindows[alert.cameraId]) {
      // Close the existing window first
      window.electron.closeAlertWindow(activeAlertWindows[alert.cameraId]);
      delete activeAlertWindows[alert.cameraId];
    }
    
    // Generate HTML content for the alert
    const htmlContent = generateAlertHTML(alert, camera);
    
    // Open a new window via Electron
    window.electron.openAlertWindow({
      title: `Alert: ${alert.objectType} on ${camera ? camera.name : `Camera ${alert.cameraId}`}`,
      content: htmlContent,
      width: 550,
      height: 600
    }).then(result => {
      if (result.success) {
        // Store the window ID
        activeAlertWindows[alert.cameraId] = result.windowId;
        
        // Auto-close after 20 seconds
        setTimeout(() => {
          closeAlertWindow(alert.cameraId);
        }, 20000);
      } else {
        console.error('Failed to open alert window:', result.error);
      }
    });
    
    // Play beep sound
    playAlertSound();
    
    return true;
  } catch (error) {
    console.error('Error opening alert window:', error);
    return false;
  }
};

// Close a specific alert window
export const closeAlertWindow = (cameraId) => {
  try {
    if (activeAlertWindows[cameraId] && window.electron && window.electron.closeAlertWindow) {
      window.electron.closeAlertWindow(activeAlertWindows[cameraId]);
      delete activeAlertWindows[cameraId];
    }
  } catch (error) {
    console.error('Error closing alert window:', error);
  }
};

// Close all alert windows
export const closeAllAlertWindows = () => {
  try {
    Object.keys(activeAlertWindows).forEach(cameraId => {
      closeAlertWindow(cameraId);
    });
  } catch (error) {
    console.error('Error closing all alert windows:', error);
  }
};

// Play alert sound
export const playAlertSound = () => {
  try {
    if (window.electron && window.electron.playBeep) {
      // Use Electron's native beep
      window.electron.playBeep();
    } else {
      // Fallback: Create and play a beep sound if electron API is not available
      const context = new (window.AudioContext || window.webkitAudioContext)();
      const oscillator = context.createOscillator();
      const gainNode = context.createGain();
      
      oscillator.type = 'sine';
      oscillator.frequency.value = 800; // Higher frequency for alert
      oscillator.connect(gainNode);
      gainNode.connect(context.destination);
      
      // Start beep
      gainNode.gain.value = 0.5;
      oscillator.start();
      
      // Stop beep after 400ms
      setTimeout(() => {
        gainNode.gain.setValueAtTime(gainNode.gain.value, context.currentTime);
        gainNode.gain.exponentialRampToValueAtTime(0.0001, context.currentTime + 0.5);
        
        // Stop oscillator after fade out
        setTimeout(() => {
          oscillator.stop();
        }, 500);
      }, 400);
    }
  } catch (error) {
    console.error('Error playing alert sound:', error);
  }
};

// Export default configuration
export const initAlertWindowManager = () => {
  // Store camera list in window object for reference
  if (!window.cameraList && window.cameras) {
    window.cameraList = window.cameras;
  }
  
  // Clean up on window unload
  window.addEventListener('beforeunload', () => {
    closeAllAlertWindows();
  });
  
  console.log('Alert Window Manager initialized');
};