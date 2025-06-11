const { app, BrowserWindow, ipcMain, dialog, shell } = require('electron');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');
const isDev = require('electron-is-dev');
const http = require('http');

let mainWindow;
let backendProcess; // Store backend process
let backendReady = false;
const alertWindows = {}; // Store alert windows

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.js'),
      webSecurity: false,
      allowRunningInsecureContent: true,
    },
  });

  // Initially load a loading page
  mainWindow.loadURL(
    isDev
      ? 'http://localhost:3001/loading.html'
      : `file://${path.join(__dirname, '../build/loading.html')}`
  );

  if (isDev) {
    mainWindow.webContents.openDevTools();
  }

  mainWindow.on('closed', () => {
    // Close all alert windows when main window is closed
    Object.values(alertWindows).forEach(win => {
      if (!win.isDestroyed()) {
        win.close();
      }
    });
    mainWindow = null;
  });

  mainWindow.webContents.on('did-fail-load', (event, errorCode, errorDescription) => {
    console.error('Electron failed to load page:', errorCode, errorDescription);
  });
checkBackendConnection()
  // Start FastAPI Backend
  // startBackend();
}

function startBackend() {
  console.log('Starting FastAPI backend...');

  const backendPath = path.join(__dirname, '../../services/main.py'); // Adjust if needed
  const venvPath = path.join(__dirname, '../../services/env'); // Adjust your virtual environment path

  let command, args;

  if (process.platform === 'win32') {
    // Windows: Use activate.bat
    command = path.join(venvPath, 'Scripts', 'python.exe'); 
    args = [backendPath];
  } else {
    // macOS/Linux: Use activate script
    command = path.join(venvPath, 'bin', 'python');
    args = [backendPath];
  }

  backendProcess = spawn(command, args);

  backendProcess.stdout.on('data', (data) => {
    console.log(`Backend Output: ${data}`);
    // Check if the backend is ready (typically when it logs "Application startup complete")
    // if (data.toString().includes('Application startup complete') || data.toString().includes('Uvicorn running')) {
      // checkBackendConnection();
    // }
  });
  setTimeout(() => {
    checkBackendConnection();

  },8000); 

  backendProcess.stderr.on('data', (data) => {
    console.error(`Backend Error: ${data}`);
  });

  backendProcess.on('close', (code) => {
    console.log(`Backend process exited with code ${code}`);
  });
}

// Check if the backend API is responsive
function checkBackendConnection() {
  console.log('Checking backend connection...');
  
  // Try to connect to the backend API
    http.get('http://localhost:8001/api/health', (res) => {
      if (res.statusCode === 200) {
        console.log('Backend is ready!');
        backendReady = true;
        
        // Load the main application
        if (mainWindow) {
          mainWindow.loadURL(
            isDev
              ? 'http://localhost:3001'
              : `file://${path.join(__dirname, '../build/index.html')}`
          );
        }
      }
    }).on('error', (err) => {
      checkBackendConnection()
      console.log('Backend not ready yet, retrying...', err.message);
    });
  // Check every second
  
  // Set a timeout in case the backend never becomes ready
  // setTimeout(() => {
  //   if (!backendReady) {
  //     clearInterval(checkInterval);
  //     console.error('Backend failed to start within timeout period');
  //     if (mainWindow) {
  //       mainWindow.webContents.send('backend-error', 'Failed to connect to AI services');
  //     }
  //   }
  // }, 30000); // 30 seconds timeout
}

// IPC handler for loading camera configuration
ipcMain.handle('load-camera-config', async () => {
  try {
    const configPath = path.join(app.getPath('userData'), 'camera.json');
    if (!fs.existsSync(configPath)) {
      const defaultConfig = { cameras: [] };
      fs.writeFileSync(configPath, JSON.stringify(defaultConfig, null, 2));
      return defaultConfig;
    }
    const data = fs.readFileSync(configPath, 'utf8');
    return JSON.parse(data);
  } catch (error) {
    console.error('Failed to load camera config:', error);
    return { cameras: [] };
  }
});

// IPC handler for saving camera configuration
ipcMain.handle('save-camera-config', async (event, config) => {
  try {
    const configPath = path.join(app.getPath('userData'), 'camera.json');
    fs.writeFileSync(configPath, JSON.stringify(config, null, 2));
    return { success: true };
  } catch (error) {
    console.error('Failed to save camera config:', error);
    return { success: false, error: error.message };
  }
});

// IPC handler for selecting output directory for recordings
ipcMain.handle('select-output-directory', async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ['openDirectory'],
  });

  if (!result.canceled) {
    return result.filePaths[0];
  }
  return null;
});

// IPC handler to check if backend is ready
ipcMain.handle('is-backend-ready', () => {
  return backendReady;
});

// IPC handler for opening alert windows
ipcMain.handle('open-alert-window', async (event, options) => {
  try {
    const { title, content, width, height } = options;
    const windowId = `alert-${Date.now()}`;
    
    // Create temp HTML file for the alert window content
    const tempDir = path.join(app.getPath('temp'), 'camera-alerts');
    if (!fs.existsSync(tempDir)) {
      fs.mkdirSync(tempDir, { recursive: true });
    }
    
    const htmlPath = path.join(tempDir, `${windowId}.html`);
    fs.writeFileSync(htmlPath, content, 'utf8');
    
    // Create new alert window
    const alertWindow = new BrowserWindow({
      width: width || 600,
      height: height || 400,
      parent: mainWindow,
      modal: false,
      alwaysOnTop: true,
      autoHideMenuBar: true,
      title: title || 'Detection Alert',
      icon: path.join(__dirname, 'alert-icon.png'), // Ensure you have this icon
      webPreferences: {
        nodeIntegration: false,
        contextIsolation: true,
        webSecurity: false,
      }
    });
    
    // Load the HTML content
    alertWindow.loadFile(htmlPath);
    
    // Store the window for later reference
    alertWindows[windowId] = alertWindow;
    
    // Play the beep sound
    shell.beep();
    
    // Clean up when window is closed
    alertWindow.on('closed', () => {
      // Remove window from registry
      delete alertWindows[windowId];
      
      // Clean up temp file
      try {
        if (fs.existsSync(htmlPath)) {
          fs.unlinkSync(htmlPath);
        }
      } catch (err) {
        console.error('Error cleaning up temp HTML file:', err);
      }
    });
    
    return { success: true, windowId };
  } catch (error) {
    console.error('Error opening alert window:', error);
    return { success: false, error: error.message };
  }
});

// IPC handler for closing alert window
ipcMain.handle('close-alert-window', async (event, windowId) => {
  try {
    const alertWindow = alertWindows[windowId];
    if (alertWindow && !alertWindow.isDestroyed()) {
      alertWindow.close();
    }
    return { success: true };
  } catch (error) {
    console.error('Error closing alert window:', error);
    return { success: false, error: error.message };
  }
});

// IPC handler for playing beep sound
ipcMain.handle('play-beep', async () => {
  try {
    shell.beep();
    return { success: true };
  } catch (error) {
    console.error('Error playing beep sound:', error);
    return { success: false, error: error.message };
  }
});

// IPC handlers for alert data persistence
ipcMain.handle('save-alert-data', async (event, data) => {
  try {
    const alertsPath = path.join(app.getPath('userData'), 'alerts.json');
    fs.writeFileSync(alertsPath, JSON.stringify(data, null, 2));
    return { success: true };
  } catch (error) {
    console.error('Failed to save alert data:', error);
    return { success: false, error: error.message };
  }
});

ipcMain.handle('load-alert-data', async () => {
  try {
    const alertsPath = path.join(app.getPath('userData'), 'alerts.json');
    if (!fs.existsSync(alertsPath)) {
      return { alerts: {} };
    }
    const data = fs.readFileSync(alertsPath, 'utf8');
    return JSON.parse(data);
  } catch (error) {
    console.error('Failed to load alert data:', error);
    return { alerts: {} };
  }
});

ipcMain.handle('clear-alert-data', async () => {
  try {
    const alertsPath = path.join(app.getPath('userData'), 'alerts.json');
    if (fs.existsSync(alertsPath)) {
      fs.unlinkSync(alertsPath);
    }
    return { success: true };
  } catch (error) {
    console.error('Failed to clear alert data:', error);
    return { success: false, error: error.message };
  }
});

// Initialize the Electron app
app.whenReady().then(createWindow);

// Quit when all windows are closed (except on macOS)
app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    if (backendProcess) {
      backendProcess.kill(); // Stop FastAPI backend
    }
    app.quit();
  }
});

// Re-create a window when the app is activated (macOS)
app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});