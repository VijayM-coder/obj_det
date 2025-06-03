// --------------- src/components/CameraView.js ---------------
import React, { useState, useEffect, useRef } from 'react';
import { connectToCamera, disconnectFromCamera } from '../services/cameraService';
import { startObjectDetection, stopObjectDetection } from '../services/detectionService';
import { startRecording, stopRecording } from '../services/recordingService.js';
import './CameraView.css';

const CameraView = ({ camera, isTracking, selectedModel, outputDir }) => {
  const [isConnected, setIsConnected] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [zoomLevel, setZoomLevel] = useState(1);
  const [detections, setDetections] = useState([]);
  const [alerts, setAlerts] = useState([]);
  
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  
  // Connect to camera on mount
  useEffect(() => {
    const connect = async () => {
      try {
        const stream = await connectToCamera(camera);
        if (stream) {
          streamRef.current = stream;
          if (videoRef.current) {
            videoRef.current.srcObject = stream;
            setIsConnected(true);
          }
        }
      } catch (error) {
        console.error(`Failed to connect to camera ${camera.id}:`, error);
      }
    };
    
    connect();
    
    // Cleanup on unmount
    return () => {
      if (streamRef.current) {
        disconnectFromCamera(streamRef.current);
        streamRef.current = null;
        setIsConnected(false);
      }
      
      if (isTracking) {
        stopObjectDetection(camera.id);
      }
      
      if (isRecording) {
        stopRecording(camera.id);
        setIsRecording(false);
      }
    };
  }, [camera]);
  
  // Handle tracking state changes
  useEffect(() => {
    if (isConnected) {
      if (isTracking) {
        startObjectDetection(camera.id, selectedModel, (results) => {
          setDetections(results.detections || []);
          
          // Check for high confidence detections (>0.7)
          const highConfidenceDetections = results.detections?.filter(d => d.confidence > 0.7) || [];
          if (highConfidenceDetections.length > 0) {
            const newAlert = {
              id: Date.now(),
              message: `High confidence detection: ${highConfidenceDetections.map(d => d.class).join(', ')}`,
              timestamp: new Date().toLocaleTimeString()
            };
            setAlerts(prev => [...prev.slice(-4), newAlert]);
          }
        });
      } else {
        stopObjectDetection(camera.id);
        setDetections([]);
      }
    }
  }, [isConnected, isTracking, camera.id, selectedModel]);
  
  // Draw detections on canvas
  useEffect(() => {
    if (canvasRef.current && detections.length > 0 && videoRef.current) {
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      const video = videoRef.current;
      
      // Match canvas size to video
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      
      // Clear previous drawings
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
      // Draw bounding boxes
      detections.forEach(detection => {
        const { bbox, class: label, confidence } = detection;
        const [x, y, width, height] = bbox;
        
        // Box style based on confidence
        ctx.strokeStyle = confidence > 0.7 ? '#FF0000' : '#00FF00';
        ctx.lineWidth = 2;
        ctx.strokeRect(x, y, width, height);
        
        // Label background
        ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
        ctx.fillRect(x, y - 20, 100, 20);
        
        // Label text
        ctx.fillStyle = '#FFFFFF';
        ctx.font = '12px Arial';
        ctx.fillText(`${label} ${Math.round(confidence * 100)}%`, x + 5, y - 5);
      });
    }
  }, [detections]);
  
  // Toggle recording
  const toggleRecording = () => {
    if (isRecording) {
      stopRecording(camera.id);
      setIsRecording(false);
    } else {
      if (outputDir) {
        startRecording(camera.id, outputDir);
        setIsRecording(true);
      } else {
        alert('Please select an output directory first');
      }
    }
  };
  
  // Zoom controls
  const handleZoomIn = () => {
    setZoomLevel(prev => Math.min(prev + 0.2, 3));
  };
  
  const handleZoomOut = () => {
    setZoomLevel(prev => Math.max(prev - 0.2, 1));
  };
  
  const handleZoomReset = () => {
    setZoomLevel(1);
  };
  
  // Apply zoom effect
  const zoomStyle = {
    transform: `scale(${zoomLevel})`,
    transformOrigin: 'center center'
  };
  
  return (
    <div className="camera-view">
      <div className="camera-header">
        <h3>{camera.name || `Camera ${camera.id}`}</h3>
        <div className="camera-controls">
          <button onClick={handleZoomIn} title="Zoom In">+</button>
          <button onClick={handleZoomReset} title="Reset Zoom">R</button>
          <button onClick={handleZoomOut} title="Zoom Out">-</button>
          <button 
            onClick={toggleRecording} 
            className={isRecording ? 'recording' : ''}
            title={isRecording ? 'Stop Recording' : 'Start Recording'}
          >
            {isRecording ? '■' : '●'}
          </button>
        </div>
      </div>
      
      <div className="video-container">
        <div className="video-wrapper" style={zoomStyle}>
          <video 
            ref={videoRef} 
            autoPlay 
            muted 
            playsInline
          />
          <canvas 
            ref={canvasRef} 
            className="detection-overlay"
          />
        </div>
        
        {alerts.length > 0 && (
          <div className="alerts-container">
            {alerts.map(alert => (
              <div key={alert.id} className="alert">
                <span className="alert-time">{alert.timestamp}</span>
                <span className="alert-message">{alert.message}</span>
              </div>
            ))}
          </div>
        )}
        
        {!isConnected && (
          <div className="connection-error">
            <p>Unable to connect to camera</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default CameraView;
