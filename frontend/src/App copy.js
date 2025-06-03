// Updated App.js - Enhanced with Alert Popup System
import React, { useState, useEffect, useCallback, useRef } from 'react';
import CameraGrid from './components/CameraGrid';
import ControlPanel from './components/ControlPanel';
import EnhancedAlertsPanel from './components/EnhancedAlertsPanel'; // Updated import
import AlertPopup from './components/AlertPopup'; // New import
import NotificationSystem from './components/NotificationSystem';
import DetectionStatus from './components/DetectionStatus';
import AlertToast from './components/AlertToast';
import AlertHistory from './components/AlertHistory';
import SecurityDashboard from './components/SecurityDashboard';
import StatusBar from './components/StatusBar';
import { getCameras, startTracking, stopTracking } from './services/apiService';
import { 
  connectToAlertWebSocket, 
  disconnectAlertWebSocket,
  subscribeToWebSocketStatus, 
  subscribeToAlerts,
  alertManager
} from './services/alertServices';
import { 
  showAlertPopup,
  subscribeToPopupEvents,
  updatePopupSettings,
  getPopupSettings
} from './services/alertPopupManager'; // New import
import { detectionMonitor } from './services/detectionMonitor';
import './App.css';

function App() {
  const [cameras, setCameras] = useState([]);
  const [cameraModelPairs, setCameraModelPairs] = useState([]);
  const [trackingPairs, setTrackingPairs] = useState([]); 
  const [isTracking, setIsTracking] = useState(false);
  const [gridLayout, setGridLayout] = useState({ rows: 2, cols: 2 });
  const [outputDir, setOutputDir] = useState('');
  const [apiStatus, setApiStatus] = useState({ isHealthy: true, lastCheck: Date.now() });
  const [activeAlertConnections, setActiveAlertConnections] = useState({});
  
  // Panel visibility states
  const [showControlPanel, setShowControlPanel] = useState(true);
  const [showDashboard, setShowDashboard] = useState(false);
  
  //add camera state
    const [showCamera, setShowCamera] = useState(false);

  const [showAlerts, setShowAlerts] = useState(false);
  const [showHistory, setShowHistory] = useState(false);
  const [showDetectionStatus, setShowDetectionStatus] = useState(false);
  
  // Alert states
  const [alertCount, setAlertCount] = useState(0);
  const [criticalAlertCount, setCriticalAlertCount] = useState(0);
  const [latestAlert, setLatestAlert] = useState(null);
  const [activeToasts, setActiveToasts] = useState([]);
  const [alertedCameras, setAlertedCameras] = useState(new Set());
  
  // Popup states - NEW
  const [activePopups, setActivePopups] = useState(new Map());
  const [popupSettings, setPopupSettings] = useState({
    enableSound: true,
    enablePopups: true,
    autoCloseDelay: 15000,
    severityFilters: {
      critical: true,
      warning: true,
      info: true
    }
  });
  
  // Settings
  const [soundEnabled, setSoundEnabled] = useState(true);
  const [flashEnabled, setFlashEnabled] = useState(true);
  const [layoutMode, setLayoutMode] = useState('standard');

  const alertsPanelRef = useRef(null);
  const audioContextRef = useRef(null);

  // Initialize audio context
  useEffect(() => {
    try {
      audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)();
    } catch (error) {
      console.warn('Audio context not available:', error);
    }
  }, []);

  // Load popup settings on startup - NEW
  useEffect(() => {
    const loadedSettings = getPopupSettings();
    setPopupSettings(loadedSettings);
    
    // Sync with existing sound settings
    updatePopupSettings({
      ...loadedSettings,
      enableSound: soundEnabled
    });
  }, []);

  // Update popup settings when sound settings change - NEW
  useEffect(() => {
    updatePopupSettings({
      ...popupSettings,
      enableSound: soundEnabled
    });
  }, [soundEnabled]);

  // Subscribe to popup events - NEW
  useEffect(() => {
    const unsubscribe = subscribeToPopupEvents((event, data) => {
      switch (event.type || event) {
        case 'show-popup':
          setActivePopups(prev => {
            const newPopups = new Map(prev);
            newPopups.set(data.alert.id, {
              alert: data.alert,
              camera: data.camera,
              timestamp: Date.now()
            });
            return newPopups;
          });
          break;
          
        case 'close-popup':
          setActivePopups(prev => {
            const newPopups = new Map(prev);
            newPopups.delete(data.alertId);
            return newPopups;
          });
          break;
          
        case 'alert-acknowledged':
          console.log('Alert acknowledged:', data.alertId);
          break;
      }
    });

    return unsubscribe;
  }, []);

  // Enhanced alert severity classification
  const classifyAlertSeverity = (alert) => {
    const confidence = alert.confidence;
    const objectType = alert.objectType.toLowerCase();
    
    if (objectType === 'person' && confidence >= 0.85) {
      return 'critical';
    }
    
    if ((objectType === 'person' && confidence >= 0.7) || 
        ['car', 'truck', 'motorcycle'].includes(objectType)) {
      return 'warning';
    }
    
    return 'info';
  };

  // Play custom alert sounds based on severity
  const playAlertSound = useCallback((severity) => {
    if (!soundEnabled || !audioContextRef.current) return;

    try {
      const context = audioContextRef.current;
      const oscillator = context.createOscillator();
      const gainNode = context.createGain();
      
      oscillator.connect(gainNode);
      gainNode.connect(context.destination);
      
      switch (severity) {
        case 'critical':
          oscillator.frequency.value = 800;
          gainNode.gain.value = 0.7;
          oscillator.start();
          oscillator.stop(context.currentTime + 0.2);
          
          setTimeout(() => {
            const osc2 = context.createOscillator();
            const gain2 = context.createGain();
            osc2.connect(gain2);
            gain2.connect(context.destination);
            osc2.frequency.value = 800;
            gain2.gain.value = 0.7;
            osc2.start();
            osc2.stop(context.currentTime + 0.2);
          }, 300);
          break;
          
        case 'warning':
          oscillator.frequency.value = 600;
          gainNode.gain.value = 0.5;
          oscillator.start();
          oscillator.stop(context.currentTime + 0.3);
          break;
          
        default:
          oscillator.frequency.value = 400;
          gainNode.gain.value = 0.3;
          oscillator.start();
          oscillator.stop(context.currentTime + 0.1);
      }
    } catch (error) {
      console.error('Error playing alert sound:', error);
    }
  }, [soundEnabled]);

  // Flash camera when alert detected
  const flashCamera = useCallback((cameraId) => {
    if (!flashEnabled) return;

    setAlertedCameras(prev => new Set([...prev, cameraId]));
    
    setTimeout(() => {
      setAlertedCameras(prev => {
        const newSet = new Set(prev);
        newSet.delete(cameraId);
        return newSet;
      });
    }, 3000);
  }, [flashEnabled]);

  // Check backend health periodically
  const checkBackendHealth = useCallback(async () => {
    try {
      const response = await fetch('http://localhost:8001/api/health', { 
        signal: AbortSignal.timeout(3000)
      });
      
      const isNowHealthy = response.ok;
      setApiStatus(prev => ({
        isHealthy: isNowHealthy,
        lastCheck: Date.now(),
        error: isNowHealthy ? undefined : prev.error
      }));
      
      if (!apiStatus.isHealthy && isNowHealthy) {
        console.log('🔄 Backend recovered, reloading cameras and reconnecting...');
        loadCameras();
        
        trackingPairs.forEach(pair => {
          console.log(`🔌 Reconnecting alert WebSocket for camera ${pair.cameraId}`);
          connectToAlertWebSocket(pair.cameraId);
        });
      }
    } catch (error) {
      console.error('Backend health check failed:', error);
      setApiStatus({
        isHealthy: false,
        lastCheck: Date.now(),
        error: error.message
      });
    }
  }, [apiStatus.isHealthy, trackingPairs]);
  
  useEffect(() => {
    const healthInterval = setInterval(checkBackendHealth, 10000);
    checkBackendHealth();
    return () => clearInterval(healthInterval);
  }, [checkBackendHealth]);
  
  const loadCameras = async () => {
    try {
      console.log('📹 Loading cameras...');
      const data = await getCameras();
      setCameras(data);
      console.log(`✅ Loaded ${data.length} cameras`);
    } catch (error) {
      console.error('❌ Failed to load cameras:', error);
      setApiStatus({
        isHealthy: false,
        lastCheck: Date.now(),
        error: 'Failed to load cameras'
      });
    }
  };
  
  // Enhanced alert subscription with popup integration - UPDATED
  useEffect(() => {
    let unsubscribe = null;
    
    if (isTracking) {
      console.log('🔔 Setting up enhanced alert subscriptions for tracking mode');
      
      unsubscribe = subscribeToAlerts('all', (newAlert) => {
        console.log('🚨 NEW ALERT RECEIVED:', {
          camera: newAlert.cameraId,
          object: newAlert.objectType,
          confidence: Math.round(newAlert.confidence * 100) + '%',
          timestamp: newAlert.timestamp
        });
        
        const severity = classifyAlertSeverity(newAlert);
        const enhancedAlert = {
          ...newAlert,
          severity,
          id: newAlert.id || `${newAlert.cameraId}_${Date.now()}_${Math.random()}`
        };
        
        setLatestAlert(enhancedAlert);
        
        if (!showAlerts) {
          setAlertCount(prev => prev + 1);
          if (severity === 'critical') {
            setCriticalAlertCount(prev => prev + 1);
          }
        }
        
        // Show popup alert - NEW
        const camera = cameras.find(cam => cam.id === newAlert.cameraId);
        showAlertPopup(enhancedAlert, camera);
        
        const toast = {
          id: enhancedAlert.id,
          alert: enhancedAlert,
          timestamp: Date.now()
        };
        
        setActiveToasts(prev => [...prev.slice(-2), toast]);
        
        setTimeout(() => {
          setActiveToasts(prev => prev.filter(t => t.id !== toast.id));
        }, 5000);
        
        playAlertSound(severity);
        flashCamera(newAlert.cameraId);
        
        if (window.electron) {
          window.electron.playBeep().catch(err => 
            console.warn('Could not play beep sound:', err)
          );
        }
      });
    }
    
    return () => {
      if (unsubscribe) {
        console.log('🔕 Cleaning up alert subscriptions');
        unsubscribe();
      }
    };
  }, [isTracking, showAlerts, cameras, playAlertSound, flashCamera]);
  
  // Subscribe to WebSocket status changes
  useEffect(() => {
    const unsubscribe = subscribeToWebSocketStatus(({ cameraId, status }) => {
      console.log(`🔌 WebSocket status for camera ${cameraId}: ${status}`);
      
      setActiveAlertConnections(prev => {
        const updated = { ...prev };
        
        if (status === 'connected') {
          updated[cameraId] = true;
        } else if (['closed', 'failed', 'disconnected', 'error'].includes(status)) {
          delete updated[cameraId];
        }
        
        return updated;
      });
    });
    
    return () => {
      unsubscribe();
    };
  }, []);
  
  useEffect(() => {
    loadCameras();
    
    return () => {
      console.log('🧹 App cleanup: Disconnecting all services...');
      alertManager.cleanup();
      detectionMonitor.stopMonitoring();
    };
  }, []);

  // Camera-model pair management
  const addCameraModelPair = (pair) => {
    console.log('➕ Adding camera-model pair:', pair);
    setCameraModelPairs(prevPairs => [...prevPairs, pair]);
  };

  const removeCameraModelPair = (cameraId) => {
    console.log('➖ Removing camera-model pair for camera:', cameraId);
    setCameraModelPairs(prevPairs => 
      prevPairs.filter(pair => pair.camera.id !== cameraId)
    );
    
    const wasTracking = trackingPairs.some(tp => tp.cameraId === cameraId);
    if (wasTracking) {
      console.log(`🔌 Disconnecting WebSocket for removed camera ${cameraId}`);
      disconnectAlertWebSocket(cameraId);
    }
  };

  // Enhanced tracking controls
  const handleStartTracking = async () => {
    try {
      if (cameraModelPairs.length === 0) {
        console.warn("⚠️ No camera-model pairs selected for tracking.");
        return;
      }
      
      console.log('🎯 Starting enhanced tracking system:', cameraModelPairs.map(p => ({
        camera: p.camera.id,
        model: p.model
      })));
      
      const updatedTrackingPairs = [];
      const failedCameras = [];

      for (const pair of cameraModelPairs) {
        try {
          console.log(`🚀 Starting tracking for camera ${pair.camera.id} with model ${pair.model}`);
          
          await startTracking(pair.camera.id, pair.model);
          
          updatedTrackingPairs.push({
            cameraId: pair.camera.id,
            model: pair.model
          });
          
          console.log(`🔗 Connecting alert WebSocket for camera ${pair.camera.id}`);
          connectToAlertWebSocket(pair.camera.id);
          
        } catch (error) {
          console.error(`❌ Failed to start tracking for camera ${pair.camera.id}:`, error);
          failedCameras.push(pair.camera.id);
        }
      }
      
      if (updatedTrackingPairs.length > 0) {
        setTrackingPairs(updatedTrackingPairs);
        setIsTracking(true);
        setShowDashboard(true);
        
        setAlertCount(0);
        setCriticalAlertCount(0);
        
        console.log('✅ Enhanced tracking started successfully for cameras:', 
          updatedTrackingPairs.map(p => p.cameraId));
      }
      
      if (failedCameras.length > 0) {
        console.error('❌ Failed to start tracking for cameras:', failedCameras);
      }

    } catch (error) {
      console.error('❌ Failed to start tracking:', error);
    }
  };

  const handleStopTracking = async () => {
    try {
      if (trackingPairs.length === 0) {
        console.warn("⚠️ No cameras are being tracked.");
        return;
      }
      
      console.log('⏹️ Stopping enhanced tracking system for cameras:', trackingPairs.map(p => p.cameraId));
      
      const failedCameras = [];
      
      for (const pair of trackingPairs) {
        try {
          console.log(`🛑 Stopping tracking for camera ${pair.cameraId}`);
          
          await stopTracking(pair.cameraId);
          
          console.log(`🔌 Disconnecting alert WebSocket for camera ${pair.cameraId}`);
          disconnectAlertWebSocket(pair.cameraId);
          
        } catch (error) {
          console.error(`❌ Failed to stop tracking for camera ${pair.cameraId}:`, error);
          failedCameras.push(pair.cameraId);
        }
      }
      
      if (failedCameras.length === 0) {
        setTrackingPairs([]);
        setIsTracking(false);
        setAlertCount(0);
        setCriticalAlertCount(0);
        setActiveToasts([]);
        setAlertedCameras(new Set());
        setActivePopups(new Map()); // Clear active popups - NEW
        console.log('✅ Enhanced tracking stopped successfully for all cameras');
      } else if (failedCameras.length < trackingPairs.length) {
        setTrackingPairs(prev => prev.filter(pair => failedCameras.includes(pair.cameraId)));
        console.error('⚠️ Failed to stop tracking for some cameras:', failedCameras);
      } else {
        console.error("❌ Failed to stop tracking for all cameras");
      }

    } catch (error) {
      console.error('❌ Failed to stop tracking:', error);
    }
  };

  const selectOutputDirectory = async () => {
    const dir = await window.electron.selectOutputDirectory();
    if (dir) {
      setOutputDir(dir);
      console.log('📁 Output directory selected:', dir);
    }
  };

  const updateGridLayout = (rows, cols) => {
    setGridLayout({ rows, cols });
    console.log(`📐 Grid layout updated: ${rows}x${cols}`);
  };

  // Panel toggles
  const togglePanel = (panel) => {
    switch (panel) {
      case 'camera':
        setShowCamera(prev => !prev);
        break;
      case 'control':
        setShowControlPanel(prev => !prev);
        break;
      case 'dashboard':
        setShowDashboard(prev => !prev);
        setShowCamera(prev => !prev);

        break;
      case 'alerts':
        setShowAlerts(prev => !prev);
        if (!showAlerts) {
          setAlertCount(0);
          setCriticalAlertCount(0);
        }
        break;
      case 'history':
        setShowHistory(prev => !prev);
        break;
      case 'detection':
        setShowDetectionStatus(prev => !prev);
        break;
      default:
        break;
    }
  };

  // Handle view alert from toast
  const handleViewAlert = (alert) => {
    if (!showAlerts) {
      togglePanel('alerts');
    }
    
    setAlertCount(0);
    setCriticalAlertCount(0);
    
    if (alert && alertsPanelRef.current) {
      console.log('👁️ Viewing specific alert:', alert);
    }
  };

  // Dismiss toast
  const dismissToast = (toastId) => {
    setActiveToasts(prev => prev.filter(t => t.id !== toastId));
  };

  // Acknowledge alert
  const acknowledgeAlert = (alertId) => {
    console.log('✅ Alert acknowledged:', alertId);
    dismissToast(alertId);
  };

  // Handle popup close - NEW
  const handlePopupClose = (alertId) => {
    setActivePopups(prev => {
      const newPopups = new Map(prev);
      newPopups.delete(alertId);
      return newPopups;
    });
  };

  // Handle popup view history - NEW
  const handlePopupViewHistory = () => {
    if (!showHistory) {
      togglePanel('history');
    }
  };

  // Get layout classes based on current layout mode
  const getLayoutClasses = () => {
    const classes = ['app-container'];
    classes.push(`layout-${layoutMode}`);
    if (!showControlPanel) classes.push('control-panel-hidden');
    return classes.join(' ');
  };

  return (
    <div className={getLayoutClasses()}>
      {/* Alert Popups - NEW */}
      {Array.from(activePopups.values()).map(popupData => (
        <AlertPopup
          key={popupData.alert.id}
          alert={popupData.alert}
          camera={popupData.camera}
          onClose={handlePopupClose}
          onViewHistory={handlePopupViewHistory}
        />
      ))}

      {/* Top Navigation Bar */}
      <header className="app-header">
        <div className="header-left">
          <div className="system-branding">
            <span className="system-icon">🛡️</span>
            <span className="system-title">Security System</span>
          </div>
          
          <div className={`system-status-indicator ${isTracking ? 'active' : 'inactive'}`}>
            <span className="status-dot"></span>
            <span className="status-text">
              {isTracking ? 'Active' : 'Inactive'}
            </span>
            {isTracking && (
              <span className="connection-info">
                {Object.keys(activeAlertConnections).length}/{trackingPairs.length}
              </span>
            )}
          </div>
        </div>

        <div className="header-center">
          {criticalAlertCount > 0 && (
            <div className="critical-alert-indicator">
              <span className="critical-icon">🚨</span>
              <span className="critical-text">
                {criticalAlertCount} Critical Alert{criticalAlertCount > 1 ? 's' : ''}
              </span>
              <button 
                className="view-critical-btn"
                onClick={() => togglePanel('alerts')}
              >
                View
              </button>
            </div>
          )}
        </div>

        <div className="header-right">
          <div className="header-actions">
            
            <button 
              className={`header-action-btn ${showControlPanel ? 'active' : ''}`}
              onClick={() => togglePanel('control')}
              title="Toggle Control Panel"
            >
              ⚙️
            </button>
            <button 
              className={`header-action-btn ${showCamera ? 'active' : ''}`}
              onClick={() => togglePanel('camera')}
              title="Toggle Camera Panel"
            >
              📹
            </button>
            
            <button 
              className={`header-action-btn ${showDashboard ? 'active' : ''}`}
              onClick={() => togglePanel('dashboard')}
              title="Security Dashboard"
            >
              📊
            </button>
            
            <button 
              className={`header-action-btn ${showAlerts ? 'active' : ''}`}
              onClick={() => togglePanel('alerts')}
              title="View Alerts"
            >
              🔔
              {alertCount > 0 && (
                <span className="notification-badge">{alertCount}</span>
              )}
            </button>
            
            <button 
              className={`header-action-btn ${showHistory ? 'active' : ''}`}
              onClick={() => togglePanel('history')}
              title="Alert History"
            >
              📋
            </button>
            
            <div className="layout-mode-selector">
              <select 
                value={layoutMode} 
                onChange={(e) => setLayoutMode(e.target.value)}
                className="layout-select"
              >
                <option value="standard">Standard</option>
                <option value="compact">Compact</option>
                <option value="fullscreen">Fullscreen</option>
              </select>
            </div>
          </div>
          
          {!apiStatus.isHealthy && (
            <div className="connection-status error">
              <span className="status-icon">⚠️</span>
              <span>Connection Lost</span>
            </div>
          )}
        </div>
      </header>

      {/* Alert Toasts */}
      <div className="alert-toasts-container">
        {activeToasts.map(toast => (
          <AlertToast
            key={toast.id}
            alert={toast.alert}
            cameras={cameras}
            onView={() => handleViewAlert(toast.alert)}
            onDismiss={() => dismissToast(toast.id)}
            onAcknowledge={() => acknowledgeAlert(toast.id)}
          />
        ))}
      </div>

      {/* Main Content Area */}
      <main className="app-main">
        {/* Left Sidebar */}
        {showControlPanel && (
          <aside className="left-sidebar">
            <ControlPanel
              cameras={cameras}
              cameraModelPairs={cameraModelPairs}
              addCameraModelPair={addCameraModelPair}
              removeCameraModelPair={removeCameraModelPair}
              isTracking={isTracking}
              startTracking={handleStartTracking}
              stopTracking={handleStopTracking}
              outputDir={outputDir}
              selectOutputDirectory={selectOutputDirectory}
              updateGridLayout={updateGridLayout}
              apiStatus={apiStatus}
              onToggleAlerts={() => togglePanel('alerts')}
              onToggleHistory={() => togglePanel('history')}
              onToggleDashboard={() => togglePanel('dashboard')}
              onToggleDetectionStatus={() => togglePanel('detection')}
              alertCount={alertCount}
              criticalAlertCount={criticalAlertCount}
              activeAlertConnections={activeAlertConnections}
              showDetectionStatus={showDetectionStatus}
              soundEnabled={soundEnabled}
              setSoundEnabled={setSoundEnabled}
              flashEnabled={flashEnabled}
              setFlashEnabled={setFlashEnabled}
            />
          </aside>
        )}

        {/* Camera Grid Area */}
        {showCamera && ( <section className="camera-section">
          <CameraGrid
            cameraModelPairs={cameraModelPairs}
            trackingPairs={trackingPairs}
            gridLayout={gridLayout}
            isTrackingActive={isTracking}
            alertedCameras={alertedCameras}
            flashEnabled={flashEnabled}
          />
        </section>)}

        {/* update right sidebar to tab: 05/23/2025 */}
        {showDashboard &&(
          <div className="panel-container">
                <div className="panel-header">
                  <h3>Security Dashboard</h3>
                  <button 
                    className="panel-close-btn"
                    onClick={() => togglePanel('dashboard')}
                  >
                    ✕
                  </button>
                </div>
                <SecurityDashboard 
                  isTrackingActive={isTracking}
                  trackingPairs={trackingPairs}
                  cameras={cameras}
                  alertCount={alertCount}
                  criticalAlertCount={criticalAlertCount}
                  activeConnections={Object.keys(activeAlertConnections).length}
                />
              </div>
        )}
       

        {/* Right Sidebar for Panels */}
        {(showDashboard || showAlerts || showHistory || showDetectionStatus) && (
          <aside className="right-sidebar">
            {/* Security Dashboard */}
            {/* {showDashboard && (
              <div className="panel-container">
                <div className="panel-header">
                  <h3>Security Dashboard</h3>
                  <button 
                    className="panel-close-btn"
                    onClick={() => togglePanel('dashboard')}
                  >
                    ✕
                  </button>
                </div>
                <SecurityDashboard 
                  isTrackingActive={isTracking}
                  trackingPairs={trackingPairs}
                  cameras={cameras}
                  alertCount={alertCount}
                  criticalAlertCount={criticalAlertCount}
                  activeConnections={Object.keys(activeAlertConnections).length}
                />
              </div>
            )} */}
            
            {/* Detection Status Panel */}
            {showDetectionStatus && (
              <div className="panel-container">
                <div className="panel-header">
                  <h3>Detection Status</h3>
                  <button 
                    className="panel-close-btn"
                    onClick={() => togglePanel('detection')}
                  >
                    ✕
                  </button>
                </div>
                <DetectionStatus 
                  isTrackingActive={isTracking}
                  trackingPairs={trackingPairs}
                />
              </div>
            )}
            
            {/* Enhanced Alerts Panel - UPDATED */}
            {showAlerts && (
              <div className="panel-container" ref={alertsPanelRef}>
                <div className="panel-header">
                  <h3>Live Alerts</h3>
                  <button 
                    className="panel-close-btn"
                    onClick={() => togglePanel('alerts')}
                  >
                    ✕
                  </button>
                </div>
                <EnhancedAlertsPanel 
                  cameras={cameras} 
                  isTrackingActive={isTracking}
                />
              </div>
            )}

            {/* Alert History Panel */}
            {showHistory && (
              <div className="panel-container">
                <div className="panel-header">
                  <h3>Alert History</h3>
                  <button 
                    className="panel-close-btn"
                    onClick={() => togglePanel('history')}
                  >
                    ✕
                  </button>
                </div>
                <AlertHistory 
                  cameras={cameras}
                  isTrackingActive={isTracking}
                />
              </div>
            )}
          </aside>
        )}
      </main>

      {/* Status Bar */}
      <StatusBar 
        cameras={cameras}
        trackingPairs={trackingPairs}
        isTracking={isTracking}
        alertCount={alertCount}
        criticalAlertCount={criticalAlertCount}
        activeConnections={Object.keys(activeAlertConnections).length}
        systemUptime={isTracking ? Date.now() : null}
      />

      {/* Notification System */}
      {isTracking && (
        <NotificationSystem 
          cameras={cameras} 
          onViewAlert={handleViewAlert}
          isTrackingActive={isTracking}
        />
      )}
    </div>
  );
}

export default App;