const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electron', {
  // Existing methods
  loadCameraConfig: () => ipcRenderer.invoke('load-camera-config'),
  saveCameraConfig: (config) => ipcRenderer.invoke('save-camera-config', config),
  selectOutputDirectory: () => ipcRenderer.invoke('select-output-directory'),
  
  // New methods for backend status
  isBackendReady: () => ipcRenderer.invoke('is-backend-ready'),
  onBackendError: (callback) => {
    ipcRenderer.on('backend-error', (_, message) => callback(message));
  },
  
  // New methods for alert windows
  openAlertWindow: (options) => ipcRenderer.invoke('open-alert-window', options),
  closeAlertWindow: (windowId) => ipcRenderer.invoke('close-alert-window', windowId),
  playBeep: () => ipcRenderer.invoke('play-beep'),
  
  // Methods for alert data persistence
  saveAlertData: (data) => ipcRenderer.invoke('save-alert-data', data),
  loadAlertData: () => ipcRenderer.invoke('load-alert-data'),
  clearAlertData: () => ipcRenderer.invoke('clear-alert-data')
});