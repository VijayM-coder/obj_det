import React, { useState, useEffect } from 'react';
import './RecordingsPanel.css';

const RecordingsPanel = ({ cameras }) => {
  const [recordings, setRecordings] = useState([]);
  const [selectedCamera, setSelectedCamera] = useState('all');
  const [selectedDate, setSelectedDate] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  
  // Fetch all recordings on mount and when filters change
  useEffect(() => {
    fetchRecordings();
  }, [selectedCamera, selectedDate]);
  
  // Fetch recordings from backend
  const fetchRecordings = async () => {
    setIsLoading(true);
    setError(null);
    
    try {
      // In a real application, you would implement a backend endpoint for this
      // For now, we'll simulate it with mock data
      await new Promise(resolve => setTimeout(resolve, 1000)); // Simulate API delay
      
      // Mock recordings data (in real app, this would come from API)
      const mockRecordings = [
        {
          id: '1',
          cameraId: cameras[0]?.id || '1',
          filename: `${cameras[0]?.id || '1'}_20250515_090112.mp4`,
          timestamp: '2025-05-15T09:01:12',
          duration: 342, // in seconds
          size: 24680000 // in bytes
        },
        {
          id: '2',
          cameraId: cameras[0]?.id || '1',
          filename: `${cameras[0]?.id || '1'}_20250515_143022.mp4`,
          timestamp: '2025-05-15T14:30:22',
          duration: 158,
          size: 12450000
        },
        {
          id: '3',
          cameraId: cameras[1]?.id || '2',
          filename: `${cameras[1]?.id || '2'}_20250516_080512.mp4`,
          timestamp: '2025-05-16T08:05:12',
          duration: 631,
          size: 45720000
        }
      ];
      
      // Apply filters
      let filteredRecordings = [...mockRecordings];
      
      if (selectedCamera !== 'all') {
        filteredRecordings = filteredRecordings.filter(rec => rec.cameraId === selectedCamera);
      }
      
      if (selectedDate) {
        const dateStr = selectedDate; // YYYY-MM-DD
        filteredRecordings = filteredRecordings.filter(rec => {
          return rec.timestamp.startsWith(dateStr);
        });
      }
      
      // Sort by timestamp (newest first)
      filteredRecordings.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
      
      setRecordings(filteredRecordings);
    } catch (err) {
      console.error('Error fetching recordings:', err);
      setError('Failed to fetch recordings. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };
  
  // Format timestamp for display
  const formatTimestamp = (isoString) => {
    try {
      const date = new Date(isoString);
      return date.toLocaleString();
    } catch (e) {
      return isoString;
    }
  };
  
  // Format duration as HH:MM:SS
  const formatDuration = (seconds) => {
    const hrs = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    
    return [
      hrs.toString().padStart(2, '0'),
      mins.toString().padStart(2, '0'),
      secs.toString().padStart(2, '0')
    ].join(':');
  };
  
  // Format file size
  const formatFileSize = (bytes) => {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    if (bytes < 1024 * 1024 * 1024) return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
    return (bytes / (1024 * 1024 * 1024)).toFixed(1) + ' GB';
  };
  
  // Get camera name by ID
  const getCameraName = (cameraId) => {
    const camera = cameras.find(cam => cam.id === cameraId);
    return camera ? (camera.name || `Camera ${cameraId}`) : cameraId;
  };
  
  // Handle camera selection change
  const handleCameraChange = (e) => {
    setSelectedCamera(e.target.value);
  };
  
  // Handle date selection change
  const handleDateChange = (e) => {
    setSelectedDate(e.target.value);
  };
  
  // Download a recording (this would be connected to a real endpoint in production)
  const handleDownload = (recording) => {
    alert(`Downloading: ${recording.filename}`);
    // In a real application, use window.location.href or fetch with blob response
    // to download the file from the server
  };
  
  // Delete a recording (this would be connected to a real endpoint in production)
  const handleDelete = (recording) => {
    if (window.confirm(`Are you sure you want to delete "${recording.filename}"?`)) {
      // In a real application, you would make an API call to delete the recording
      setRecordings(prev => prev.filter(rec => rec.id !== recording.id));
    }
  };
  
  return (
    <div className="recordings-panel">
      <div className="recordings-header">
        <h2>Recorded Videos</h2>
        <div className="filter-controls">
          <div className="camera-filter">
            <label>Camera:</label>
            <select value={selectedCamera} onChange={handleCameraChange}>
              <option value="all">All Cameras</option>
              {cameras.map(camera => (
                <option key={camera.id} value={camera.id}>
                  {camera.name || `Camera ${camera.id}`}
                </option>
              ))}
            </select>
          </div>
          <div className="date-filter">
            <label>Date:</label>
            <input 
              type="date" 
              value={selectedDate} 
              onChange={handleDateChange}
            />
          </div>
          <button 
            className="refresh-button" 
            onClick={fetchRecordings}
            disabled={isLoading}
          >
            {isLoading ? 'Loading...' : 'Refresh'}
          </button>
        </div>
      </div>
      
      {error && (
        <div className="error-message">
          {error}
        </div>
      )}
      
      <div className="recordings-list">
        {recordings.length === 0 ? (
          <div className="no-recordings">
            <p>{isLoading ? 'Loading recordings...' : 'No recordings found'}</p>
          </div>
        ) : (
          <table className="recordings-table">
            <thead>
              <tr>
                <th>Camera</th>
                <th>Date & Time</th>
                <th>Duration</th>
                <th>Size</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {recordings.map(recording => (
                <tr key={recording.id}>
                  <td>{getCameraName(recording.cameraId)}</td>
                  <td>{formatTimestamp(recording.timestamp)}</td>
                  <td>{formatDuration(recording.duration)}</td>
                  <td>{formatFileSize(recording.size)}</td>
                  <td className="actions">
                    <button 
                      className="download-button"
                      onClick={() => handleDownload(recording)}
                      title="Download"
                    >
                      ⬇️
                    </button>
                    <button 
                      className="delete-button"
                      onClick={() => handleDelete(recording)}
                      title="Delete"
                    >
                      🗑️
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
};

export default RecordingsPanel;