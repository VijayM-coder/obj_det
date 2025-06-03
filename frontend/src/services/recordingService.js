// Updated src/services/recordingService.js
const API_BASE_URL = 'http://localhost:8001/api';

// Start recording for a specific camera
export const startRecording = async (cameraId, outputDir = null) => {
  try {
    const requestBody = outputDir ? { output_dir: outputDir } : {};
    
    const response = await fetch(`${API_BASE_URL}/cameras/${cameraId}/start_recording`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestBody),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
    }

    const result = await response.json();
    console.log(`✅ Recording started for camera ${cameraId}:`, result);
    
    // Store recording status locally
    if (window.electron && window.electron.saveData) {
      const activeRecordings = await getActiveRecordings();
      if (!activeRecordings.includes(cameraId)) {
        activeRecordings.push(cameraId);
        await window.electron.saveData('activeRecordings.json', { cameras: activeRecordings });
      }
    }
    
    return result;
  } catch (error) {
    console.error('❌ Error starting recording:', error);
    throw error;
  }
};

// Stop recording for a specific camera
export const stopRecording = async (cameraId) => {
  try {
    const response = await fetch(`${API_BASE_URL}/cameras/${cameraId}/stop_recording`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
    }

    const result = await response.json();
    console.log(`✅ Recording stopped for camera ${cameraId}:`, result);
    
    // Update local storage
    if (window.electron && window.electron.saveData) {
      const activeRecordings = await getActiveRecordings();
      const updatedRecordings = activeRecordings.filter(id => id !== cameraId);
      await window.electron.saveData('activeRecordings.json', { cameras: updatedRecordings });
    }
    
    return result;
  } catch (error) {
    console.error('❌ Error stopping recording:', error);
    throw error;
  }
};

// Check recording status for a camera
export const checkRecordingStatus = async (cameraId) => {
  try {
    const response = await fetch(`${API_BASE_URL}/cameras/${cameraId}/recording_status`);
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const result = await response.json();
    return result.is_recording;
  } catch (error) {
    console.error(`❌ Error checking recording status for camera ${cameraId}:`, error);
    return false;
  }
};

// Get global recording status
export const getGlobalRecordingStatus = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/recording/status`);
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('❌ Error getting global recording status:', error);
    throw error;
  }
};

// Start recording on all cameras
export const startAllRecordings = async (outputDir = null) => {
  try {
    const requestBody = outputDir ? { output_dir: outputDir } : {};
    
    const response = await fetch(`${API_BASE_URL}/recording/start_all`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestBody),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
    }

    const result = await response.json();
    console.log('✅ Started recording on all cameras:', result);
    
    // Update local storage with all started recordings
    if (window.electron && window.electron.saveData && result.started_recordings) {
      const cameraIds = result.started_recordings.map(r => r.camera_id);
      await window.electron.saveData('activeRecordings.json', { cameras: cameraIds });
    }
    
    return result;
  } catch (error) {
    console.error('❌ Error starting all recordings:', error);
    throw error;
  }
};

// Stop recording on all cameras
export const stopAllRecordings = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/recording/stop_all`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
    }

    const result = await response.json();
    console.log('✅ Stopped recording on all cameras:', result);
    
    // Clear local storage
    if (window.electron && window.electron.saveData) {
      await window.electron.saveData('activeRecordings.json', { cameras: [] });
    }
    
    return result;
  } catch (error) {
    console.error('❌ Error stopping all recordings:', error);
    throw error;
  }
};

// Get list of recordings with optional filtering
export const getRecordings = async (cameraId = null, date = null, limit = 50) => {
  try {
    const params = new URLSearchParams();
    if (cameraId) params.append('camera_id', cameraId);
    if (date) params.append('date', date);
    if (limit) params.append('limit', limit.toString());
    
    const url = `${API_BASE_URL}/recordings${params.toString() ? '?' + params.toString() : ''}`;
    const response = await fetch(url);
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('❌ Error fetching recordings:', error);
    throw error;
  }
};

// Delete a recording
export const deleteRecording = async (recordingId) => {
  try {
    const response = await fetch(`${API_BASE_URL}/recordings/${recordingId}`, {
      method: 'DELETE',
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
    }

    const result = await response.json();
    console.log(`✅ Recording ${recordingId} deleted:`, result);
    return result;
  } catch (error) {
    console.error(`❌ Error deleting recording ${recordingId}:`, error);
    throw error;
  }
};

// Download a recording
export const downloadRecording = async (recordingId, filename) => {
  try {
    const response = await fetch(`${API_BASE_URL}/recordings/${recordingId}/download`);
    
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
    }
    
    // Create download link
    const blob = await response.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.style.display = 'none';
    a.href = url;
    a.download = filename || `recording_${recordingId}.mp4`;
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);
    
    console.log(`✅ Recording downloaded: ${filename}`);
    return true;
  } catch (error) {
    console.error(`❌ Error downloading recording ${recordingId}:`, error);
    throw error;
  }
};

// Get/Set recording configuration
export const getRecordingConfig = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/recording/config`);
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('❌ Error getting recording config:', error);
    throw error;
  }
};

export const updateRecordingConfig = async (config) => {
  try {
    const response = await fetch(`${API_BASE_URL}/recording/config`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(config),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error('❌ Error updating recording config:', error);
    throw error;
  }
};

// Legacy functions for backward compatibility
export const getActiveRecordings = async () => {
  try {
    // First try to get from server
    const status = await getGlobalRecordingStatus();
    return status.active_recordings.map(r => r.camera_id);
  } catch (error) {
    // Fallback to local storage
    if (window.electron && window.electron.loadData) {
      try {
        const data = await window.electron.loadData('activeRecordings.json');
        return data?.cameras || [];
      } catch (localError) {
        console.error('❌ Error loading active recordings from local storage:', localError);
        return [];
      }
    }
    return [];
  }
};

export const getRecordingOutputDir = async () => {
  try {
    const config = await getRecordingConfig();
    return config.output_directory || '';
  } catch (error) {
    // Fallback to local storage
    if (window.electron && window.electron.loadData) {
      try {
        const data = await window.electron.loadData('recordingSettings.json');
        return data?.outputDir || '';
      } catch (localError) {
        console.error('❌ Error loading recording settings:', localError);
        return '';
      }
    }
    return '';
  }
};

export const setRecordingOutputDir = async (outputDir) => {
  try {
    // Update server config
    await updateRecordingConfig({ output_directory: outputDir });
    
    // Also save locally for backup
    if (window.electron && window.electron.saveData) {
      await window.electron.saveData('recordingSettings.json', { outputDir });
    }
    
    return true;
  } catch (error) {
    console.error('❌ Error setting recording output directory:', error);
    return false;
  }
};

export const selectOutputDirectory = async () => {
  if (window.electron && window.electron.selectOutputDirectory) {
    try {
      const dir = await window.electron.selectOutputDirectory();
      if (dir) {
        await setRecordingOutputDir(dir);
        return dir;
      }
    } catch (error) {
      console.error('❌ Error selecting output directory:', error);
    }
  }
  return null;
};

// Utility function to format recording duration
export const formatDuration = (seconds) => {
  if (!seconds) return '0:00';
  
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secs = Math.floor(seconds % 60);
  
  if (hours > 0) {
    return `${hours}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  }
  return `${minutes}:${secs.toString().padStart(2, '0')}`;
};

// Utility function to format file size
export const formatFileSize = (bytes) => {
  if (!bytes) return '0 B';
  
  const units = ['B', 'KB', 'MB', 'GB'];
  let size = bytes;
  let unitIndex = 0;
  
  while (size >= 1024 && unitIndex < units.length - 1) {
    size /= 1024;
    unitIndex++;
  }
  
  return `${size.toFixed(1)} ${units[unitIndex]}`;
};