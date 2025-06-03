// Complete RecordingManager.js - Full recording management with filtering and controls
import React, { useState, useEffect, useCallback } from 'react';
import { 
  getRecordings, 
  deleteRecording, 
  downloadRecording, 
  getGlobalRecordingStatus,
  formatDuration,
  formatFileSize,
  startRecording,
  stopRecording,
  startAllRecordings,
  stopAllRecordings
} from '../services/recordingService';
import './RecordingManager.css';

const RecordingManager = ({ cameras, isVisible, onClose, outputDir }) => {
  const [recordings, setRecordings] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selectedCamera, setSelectedCamera] = useState('');
  const [selectedDate, setSelectedDate] = useState('');
  const [selectedStatus, setSelectedStatus] = useState('');
  const [globalStatus, setGlobalStatus] = useState(null);
  const [sortBy, setSortBy] = useState('start_time');
  const [sortOrder, setSortOrder] = useState('desc');
  const [selectedRecordings, setSelectedRecordings] = useState(new Set());
  const [showBulkActions, setShowBulkActions] = useState(false);
  const [activeTab, setActiveTab] = useState('recordings'); // 'recordings' or 'controls'

  // Load recordings when component becomes visible
  useEffect(() => {
    if (isVisible) {
      loadRecordings();
      loadGlobalStatus();
      
      // Set up periodic refresh
      const interval = setInterval(() => {
        if (isVisible) {
          loadGlobalStatus();
        }
      }, 5000);
      
      return () => clearInterval(interval);
    }
  }, [isVisible, selectedCamera, selectedDate, selectedStatus]);

  const loadRecordings = useCallback(async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await getRecordings(
        selectedCamera || null, 
        selectedDate || null,
        200
      );
      
      let recordingsList = response.recordings || [];
      
      // Filter by status if selected
      if (selectedStatus) {
        recordingsList = recordingsList.filter(r => r.status === selectedStatus);
      }
      
      // Sort recordings
      recordingsList.sort((a, b) => {
        const aValue = a[sortBy];
        const bValue = b[sortBy];
        
        if (sortOrder === 'asc') {
          return aValue < bValue ? -1 : aValue > bValue ? 1 : 0;
        } else {
          return aValue > bValue ? -1 : aValue < bValue ? 1 : 0;
        }
      });
      
      setRecordings(recordingsList);
    } catch (err) {
      setError(err.message);
      console.error('Error loading recordings:', err);
    } finally {
      setLoading(false);
    }
  }, [selectedCamera, selectedDate, selectedStatus, sortBy, sortOrder]);

  const loadGlobalStatus = async () => {
    try {
      const status = await getGlobalRecordingStatus();
      setGlobalStatus(status);
    } catch (err) {
      console.error('Error loading global recording status:', err);
    }
  };

  const handleDeleteRecording = async (recordingId, filename) => {
    if (!window.confirm(`Are you sure you want to delete "${filename}"?`)) {
      return;
    }

    try {
      await deleteRecording(recordingId);
      setRecordings(prev => prev.filter(r => r.id !== recordingId));
      setSelectedRecordings(prev => {
        const newSet = new Set(prev);
        newSet.delete(recordingId);
        return newSet;
      });
      loadGlobalStatus(); // Refresh status
    } catch (err) {
      setError(`Failed to delete recording: ${err.message}`);
    }
  };

  const handleBulkDelete = async () => {
    if (selectedRecordings.size === 0) return;
    
    const recordingNames = recordings
      .filter(r => selectedRecordings.has(r.id))
      .map(r => r.filename)
      .join(', ');
      
    if (!window.confirm(`Are you sure you want to delete ${selectedRecordings.size} recordings?\n\n${recordingNames}`)) {
      return;
    }

    setLoading(true);
    const failures = [];
    
    for (const recordingId of selectedRecordings) {
      try {
        await deleteRecording(recordingId);
      } catch (err) {
        failures.push(recordingId);
      }
    }
    
    if (failures.length > 0) {
      setError(`Failed to delete ${failures.length} recordings`);
    }
    
    setSelectedRecordings(new Set());
    await loadRecordings();
    loadGlobalStatus();
  };

  const handleDownloadRecording = async (recordingId, filename) => {
    try {
      await downloadRecording(recordingId, filename);
    } catch (err) {
      setError(`Failed to download recording: ${err.message}`);
    }
  };

  const handleSort = (field) => {
    if (sortBy === field) {
      setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
    } else {
      setSortBy(field);
      setSortOrder('desc');
    }
  };

  const handleSelectRecording = (recordingId) => {
    setSelectedRecordings(prev => {
      const newSet = new Set(prev);
      if (newSet.has(recordingId)) {
        newSet.delete(recordingId);
      } else {
        newSet.add(recordingId);
      }
      return newSet;
    });
  };

  const handleSelectAll = () => {
    if (selectedRecordings.size === recordings.length) {
      setSelectedRecordings(new Set());
    } else {
      setSelectedRecordings(new Set(recordings.map(r => r.id)));
    }
  };

  const handleStartRecording = async (cameraId) => {
    try {
      await startRecording(cameraId, outputDir);
      loadGlobalStatus();
    } catch (err) {
      setError(`Failed to start recording: ${err.message}`);
    }
  };

  const handleStopRecording = async (cameraId) => {
    try {
      await stopRecording(cameraId);
      loadGlobalStatus();
      loadRecordings(); // Refresh to show new completed recording
    } catch (err) {
      setError(`Failed to stop recording: ${err.message}`);
    }
  };

  const handleStartAllRecordings = async () => {
    try {
      await startAllRecordings(outputDir);
      loadGlobalStatus();
    } catch (err) {
      setError(`Failed to start all recordings: ${err.message}`);
    }
  };

  const handleStopAllRecordings = async () => {
    try {
      await stopAllRecordings();
      loadGlobalStatus();
      loadRecordings(); // Refresh to show new completed recordings
    } catch (err) {
      setError(`Failed to stop all recordings: ${err.message}`);
    }
  };

  const getCameraName = (cameraId) => {
    const camera = cameras.find(cam => cam.id.toString() === cameraId.toString());
    return camera ? (camera.name || `Camera ${camera.id}`) : `Camera ${cameraId}`;
  };

  const getStatusBadgeClass = (status) => {
    switch (status) {
      case 'recording':
        return 'status-recording';
      case 'completed':
        return 'status-completed';
      case 'error':
        return 'status-error';
      default:
        return 'status-unknown';
    }
  };

  const formatDateTime = (dateTimeString) => {
    if (!dateTimeString) return 'N/A';
    const date = new Date(dateTimeString);
    return date.toLocaleString();
  };

  const getActiveRecordings = () => {
    return globalStatus?.active_recordings || [];
  };

  const isRecording = (cameraId) => {
    return getActiveRecordings().some(r => r.camera_id === cameraId);
  };

  if (!isVisible) return null;

  return (
    <div className="recording-manager-overlay">
      <div className="recording-manager-modal">
        <div className="recording-manager-header">
          <div className="header-left">
            <h2>Recording Manager</h2>
            {globalStatus && (
              <div className="global-status">
                <span className="status-item">
                  <span className="status-label">Total:</span>
                  <span className="status-value">{globalStatus.total_recordings_in_history}</span>
                </span>
                <span className="status-item">
                  <span className="status-label">Active:</span>
                  <span className="status-value recording">
                    {globalStatus.cameras_recording}/{globalStatus.total_cameras}
                  </span>
                </span>
                <span className="status-item">
                  <span className="status-label">Selected:</span>
                  <span className="status-value">{selectedRecordings.size}</span>
                </span>
              </div>
            )}
          </div>
          <button className="close-btn" onClick={onClose}>✕</button>
        </div>

        {/* Tab Navigation */}
        <div className="tab-navigation">
          <button 
            className={`tab-button ${activeTab === 'recordings' ? 'active' : ''}`}
            onClick={() => setActiveTab('recordings')}
          >
            📋 Recordings
          </button>
          <button 
            className={`tab-button ${activeTab === 'controls' ? 'active' : ''}`}
            onClick={() => setActiveTab('controls')}
          >
            🎛️ Controls
          </button>
        </div>

        <div className="recording-manager-content">
          {activeTab === 'recordings' && (
            <>
              {/* Filters */}
              <div className="filters-section">
                <div className="filter-row">
                  <div className="filter-group">
                    <label>Camera:</label>
                    <select 
                      value={selectedCamera} 
                      onChange={(e) => setSelectedCamera(e.target.value)}
                    >
                      <option value="">All Cameras</option>
                      {cameras.map(camera => (
                        <option key={camera.id} value={camera.id}>
                          {camera.name || `Camera ${camera.id}`}
                        </option>
                      ))}
                    </select>
                  </div>

                  <div className="filter-group">
                    <label>Date:</label>
                    <input 
                      type="date" 
                      value={selectedDate}
                      onChange={(e) => setSelectedDate(e.target.value)}
                    />
                  </div>

                  <div className="filter-group">
                    <label>Status:</label>
                    <select 
                      value={selectedStatus} 
                      onChange={(e) => setSelectedStatus(e.target.value)}
                    >
                      <option value="">All Status</option>
                      <option value="recording">Recording</option>
                      <option value="completed">Completed</option>
                      <option value="error">Error</option>
                    </select>
                  </div>

                  <div className="filter-actions">
                    <button 
                      onClick={() => {
                        setSelectedCamera('');
                        setSelectedDate('');
                        setSelectedStatus('');
                      }}
                      className="clear-filters-btn"
                    >
                      Clear
                    </button>
                    <button 
                      onClick={loadRecordings}
                      className="refresh-btn"
                      disabled={loading}
                    >
                      🔄 Refresh
                    </button>
                  </div>
                </div>

                {/* Bulk Actions */}
                {selectedRecordings.size > 0 && (
                  <div className="bulk-actions">
                    <span className="bulk-info">
                      {selectedRecordings.size} recording{selectedRecordings.size !== 1 ? 's' : ''} selected
                    </span>
                    <div className="bulk-buttons">
                      <button 
                        onClick={handleBulkDelete}
                        className="bulk-delete-btn"
                        disabled={loading}
                      >
                        🗑️ Delete Selected
                      </button>
                      <button 
                        onClick={() => setSelectedRecordings(new Set())}
                        className="bulk-clear-btn"
                      >
                        Clear Selection
                      </button>
                    </div>
                  </div>
                )}
              </div>

              {/* Error Display */}
              {error && (
                <div className="error-message">
                  <span className="error-icon">⚠️</span>
                  <span>{error}</span>
                  <button onClick={() => setError(null)} className="dismiss-error">✕</button>
                </div>
              )}

              {/* Loading */}
              {loading && (
                <div className="loading-indicator">
                  <div className="loading-spinner"></div>
                  <span>Loading recordings...</span>
                </div>
              )}

              {/* Recordings Table */}
              {!loading && recordings.length > 0 && (
                <div className="recordings-table-container">
                  <table className="recordings-table">
                    <thead>
                      <tr>
                        <th className="select-column">
                          <input
                            type="checkbox"
                            checked={selectedRecordings.size === recordings.length && recordings.length > 0}
                            onChange={handleSelectAll}
                          />
                        </th>
                        <th 
                          onClick={() => handleSort('camera_id')}
                          className={`sortable ${sortBy === 'camera_id' ? `sorted-${sortOrder}` : ''}`}
                        >
                          Camera
                        </th>
                        <th 
                          onClick={() => handleSort('filename')}
                          className={`sortable ${sortBy === 'filename' ? `sorted-${sortOrder}` : ''}`}
                        >
                          Filename
                        </th>
                        <th 
                          onClick={() => handleSort('start_time')}
                          className={`sortable ${sortBy === 'start_time' ? `sorted-${sortOrder}` : ''}`}
                        >
                          Start Time
                        </th>
                        <th 
                          onClick={() => handleSort('duration_seconds')}
                          className={`sortable ${sortBy === 'duration_seconds' ? `sorted-${sortOrder}` : ''}`}
                        >
                          Duration
                        </th>
                        <th 
                          onClick={() => handleSort('file_size_bytes')}
                          className={`sortable ${sortBy === 'file_size_bytes' ? `sorted-${sortOrder}` : ''}`}
                        >
                          Size
                        </th>
                        <th 
                          onClick={() => handleSort('status')}
                          className={`sortable ${sortBy === 'status' ? `sorted-${sortOrder}` : ''}`}
                        >
                          Status
                        </th>
                        <th>Actions</th>
                      </tr>
                    </thead>
                    <tbody>
                      {recordings.map(recording => (
                        <tr 
                          key={recording.id} 
                          className={`recording-row ${selectedRecordings.has(recording.id) ? 'selected' : ''}`}
                        >
                          <td className="select-cell">
                            <input
                              type="checkbox"
                              checked={selectedRecordings.has(recording.id)}
                              onChange={() => handleSelectRecording(recording.id)}
                            />
                          </td>
                          
                          <td className="camera-cell">
                            <div className="camera-info">
                              <span className="camera-name">
                                {getCameraName(recording.camera_id)}
                              </span>
                              <span className="camera-id">
                                ID: {recording.camera_id}
                              </span>
                            </div>
                          </td>
                          
                          <td className="filename-cell">
                            <div className="filename-info">
                              <span className="filename" title={recording.filename}>
                                {recording.filename}
                              </span>
                            </div>
                          </td>
                          
                          <td className="datetime-cell">
                            {formatDateTime(recording.start_time)}
                          </td>
                          
                          <td className="duration-cell">
                            {recording.duration_seconds ? 
                              formatDuration(recording.duration_seconds) : 
                              (recording.status === 'recording' ? 'Recording...' : 'N/A')
                            }
                          </td>
                          
                          <td className="size-cell">
                            {recording.file_size_bytes ? 
                              formatFileSize(recording.file_size_bytes) : 
                              'N/A'
                            }
                          </td>
                          
                          <td className="status-cell">
                            <span className={`status-badge ${getStatusBadgeClass(recording.status)}`}>
                              {recording.status}
                            </span>
                          </td>
                          
                          <td className="actions-cell">
                            <div className="action-buttons">
                              {recording.status === 'completed' && (
                                <button
                                  onClick={() => handleDownloadRecording(recording.id, recording.filename)}
                                  className="action-btn download-btn"
                                  title="Download Recording"
                                >
                                  ⬇️
                                </button>
                              )}
                              
                              {recording.status !== 'recording' && (
                                <button
                                  onClick={() => handleDeleteRecording(recording.id, recording.filename)}
                                  className="action-btn delete-btn"
                                  title="Delete Recording"
                                >
                                  🗑️
                                </button>
                              )}
                            </div>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}

              {/* Empty State */}
              {!loading && recordings.length === 0 && (
                <div className="empty-state">
                  <div className="empty-icon">📹</div>
                  <h3>No Recordings Found</h3>
                  <p>
                    {selectedCamera || selectedDate || selectedStatus ? 
                      'No recordings match your current filters.' : 
                      'No recordings have been made yet.'
                    }
                  </p>
                  {(selectedCamera || selectedDate || selectedStatus) && (
                    <button 
                      onClick={() => {
                        setSelectedCamera('');
                        setSelectedDate('');
                        setSelectedStatus('');
                      }}
                      className="clear-filters-btn"
                    >
                      Clear Filters
                    </button>
                  )}
                </div>
              )}
            </>
          )}

          {activeTab === 'controls' && (
            <div className="recording-controls-tab">
              {/* Global Controls */}
              <div className="global-controls-section">
                <h3>Global Recording Controls</h3>
                
                <div className="global-status-display">
                  {globalStatus && (
                    <div className="status-grid">
                      <div className="status-card">
                        <span className="status-number">{globalStatus.cameras_recording}</span>
                        <span className="status-label">Recording Now</span>
                      </div>
                      <div className="status-card">
                        <span className="status-number">{globalStatus.total_cameras}</span>
                        <span className="status-label">Total Cameras</span>
                      </div>
                      <div className="status-card">
                        <span className="status-number">{globalStatus.total_recordings_in_history}</span>
                        <span className="status-label">Total Recordings</span>
                      </div>
                    </div>
                  )}
                </div>

                <div className="global-actions">
                  <button 
                    onClick={handleStartAllRecordings}
                    disabled={!outputDir || (globalStatus?.cameras_recording === globalStatus?.total_cameras)}
                    className="global-action-btn start"
                  >
                    ⏺️ Start All Recording
                  </button>
                  
                  <button 
                    onClick={handleStopAllRecordings}
                    disabled={globalStatus?.cameras_recording === 0}
                    className="global-action-btn stop"
                  >
                    ⏹️ Stop All Recording
                  </button>
                </div>

                <div className="output-directory-info">
                  <div className="directory-display">
                    <span className="directory-label">Output Directory:</span>
                    <span className="directory-path">{outputDir || 'Not configured'}</span>
                  </div>
                  {!outputDir && (
                    <div className="directory-warning">
                      <span className="warning-icon">⚠️</span>
                      <span>Configure output directory in settings to enable recording</span>
                    </div>
                  )}
                </div>
              </div>

              {/* Individual Camera Controls */}
              <div className="individual-controls-section">
                <h3>Individual Camera Controls</h3>
                
                <div className="camera-controls-grid">
                  {cameras.map(camera => (
                    <div key={camera.id} className="camera-control-card">
                      <div className="camera-control-header">
                        <span className="camera-control-name">
                          {camera.name || `Camera ${camera.id}`}
                        </span>
                        <span className={`camera-control-status ${isRecording(camera.id) ? 'recording' : 'stopped'}`}>
                          {isRecording(camera.id) ? 'Recording' : 'Stopped'}
                        </span>
                      </div>
                      
                      <div className="camera-control-actions">
                        <button
                          onClick={() => 
                            isRecording(camera.id) 
                              ? handleStopRecording(camera.id)
                              : handleStartRecording(camera.id)
                          }
                          className={`camera-control-btn ${isRecording(camera.id) ? 'stop' : 'start'}`}
                          disabled={!outputDir}
                        >
                          {isRecording(camera.id) ? '⏹️ Stop' : '⏺️ Start'}
                        </button>
                      </div>
                      
                      {isRecording(camera.id) && (
                        <div className="recording-indicator">
                          <span className="recording-dot"></span>
                          <span>Recording active</span>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>

              {/* Recording Information */}
              <div className="recording-info-section">
                <h3>Recording Information</h3>
                <div className="recording-details-grid">
                  <div className="detail-card">
                    <span className="detail-label">Stream Types:</span>
                    <span className="detail-value">Live Stream + Detection Stream</span>
                  </div>
                  <div className="detail-card">
                    <span className="detail-label">Format:</span>
                    <span className="detail-value">MP4 (H.264)</span>
                  </div>
                  <div className="detail-card">
                    <span className="detail-label">Quality:</span>
                    <span className="detail-value">Original Resolution</span>
                  </div>
                  <div className="detail-card">
                    <span className="detail-label">Storage:</span>
                    <span className="detail-value">{outputDir || 'Not configured'}</span>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="recording-manager-footer">
          <div className="footer-info">
            {activeTab === 'recordings' && recordings.length > 0 && (
              <span>
                Showing {recordings.length} recording{recordings.length !== 1 ? 's' : ''}
                {selectedRecordings.size > 0 && ` (${selectedRecordings.size} selected)`}
              </span>
            )}
            {globalStatus && (
              <span className="output-dir">
                Output: {globalStatus.output_directory}
              </span>
            )}
          </div>
          
          <div className="footer-actions">
            <button onClick={onClose} className="close-footer-btn">
              Close
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default RecordingManager;