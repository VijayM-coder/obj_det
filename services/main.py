# Enhanced main.py with Multi-Model Support
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks, Response, APIRouter
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Any, Union
import json
import cv2
import numpy as np
import asyncio
import threading
import time
import os
import uuid
from datetime import datetime
import torch
from ultralytics import YOLO
from contextlib import asynccontextmanager
import base64
import random
import queue
import logging
from concurrent.futures import ThreadPoolExecutor
import weakref

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
CAMERA_CONFIG_PATH = "camera.json"
MODELS_DIR = "models/"
RECORDINGS_DIR = "recordings/"
PRE_RECORDING_BUFFER_SECONDS = 5
MAX_FRAME_QUEUE_SIZE = 10
DETECTION_BATCH_SIZE = 4
MAX_WORKERS = 8  # Increased for multi-model support

# Ensure directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RECORDINGS_DIR, exist_ok=True)

# Data models
class Camera(BaseModel):
    id: Union[int, str]
    name: str
    type: str  # "webcam" or "ip"
    source: str  # Either webcam index or IP camera URL
    resolution: str
    fps: int
    enabled: bool = True
    recording: bool = False
    detection_enabled: bool = True
    alert_threshold: float = 0.5

class TrackingRequest(BaseModel):
    model: str
    session_id: Optional[str] = None
    pair_id: Optional[str] = None
    scenario: Optional[str] = "multi-object"
    enable_multi_model: Optional[bool] = False

class RecordingConfig(BaseModel):
    output_directory: Optional[str] = None
    format: str = "mp4"
    quality: int = 80
    max_file_size_mb: Optional[int] = None
    max_duration_minutes: Optional[int] = None

class RecordingInfo(BaseModel):
    id: str
    camera_id: str
    filename: str
    file_path: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    file_size_bytes: Optional[int] = None
    status: str  # "recording", "completed", "error"

class DetectionSession:
    """Represents a detection session for a specific camera-model combination"""
    def __init__(self, session_id: str, camera_id: str, model_id: str, model: YOLO):
        self.session_id = session_id
        self.camera_id = camera_id
        self.model_id = model_id
        self.model = model
        self.active = True
        self.detection_queue = queue.Queue(maxsize=5)
        self.detections = []
        self.processing_thread = None
        self.websocket_clients = []
        
    def start_processing(self, camera_manager):
        """Start the detection processing thread for this session"""
        self.processing_thread = threading.Thread(
            target=self._process_detections, 
            args=(camera_manager,), 
            daemon=True
        )
        self.processing_thread.start()
        logger.info(f"Started detection processing for session {self.session_id}")
        
    def stop_processing(self):
        """Stop the detection processing"""
        self.active = False
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2)
        logger.info(f"Stopped detection processing for session {self.session_id}")
        
    def _process_detections(self, camera_manager):
        """Process detections for this specific session"""
        while self.active:
            try:
                if not self.detection_queue.empty():
                    frame, timestamp = self.detection_queue.get_nowait()
                    
                    # Run detection with this session's model
                    results = self.model.predict([frame], conf=0.25, verbose=False)
                    
                    camera = camera_manager.cameras.get(self.camera_id)
                    if not camera or not camera.detection_enabled:
                        continue
                        
                    detections = []
                    for result in results:
                        boxes = result.boxes
                        if boxes is not None:
                            for box in boxes:
                                x1, y1, x2, y2 = box.xyxy[0].tolist()
                                confidence = box.conf.item()
                                class_id = int(box.cls.item())
                                class_name = result.names[class_id]
                                
                                detection = {
                                    "bbox": [x1, y1, x2, y2],
                                    "confidence": confidence,
                                    "class_id": class_id,
                                    "class_name": class_name,
                                    "timestamp": datetime.fromtimestamp(timestamp).isoformat(),
                                    "model": self.model_id,
                                    "session_id": self.session_id
                                }
                                detections.append(detection)
                                
                                # Send alert if confidence exceeds threshold
                                if confidence > camera.alert_threshold:
                                    self._send_alert(camera_manager, detection, frame)
                    
                    # Update session detections
                    self.detections = detections
                    
                else:
                    time.sleep(0.01)  # Small delay if no frames to process
                    
            except Exception as e:
                logger.error(f"Error in detection session {self.session_id}: {e}")
                time.sleep(0.1)
                
    def _send_alert(self, camera_manager, detection, frame):
        """Send an alert for this specific session"""
        try:
            # Crop the detection area
            x1, y1, x2, y2 = map(int, detection["bbox"])
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            detection_crop = frame[y1:y2, x1:x2] if x2 > x1 and y2 > y1 else frame
            
            # Encode cropped image
            _, img_encoded = cv2.imencode('.jpg', detection_crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
            img_base64 = base64.b64encode(img_encoded.tobytes()).decode('utf-8')

            alert_data = {
                "id": f"{self.camera_id}_{self.model_id}_{int(time.time() * 1000)}_{random.randint(1000, 9999)}",
                "camera_id": self.camera_id,
                "model": self.model_id,
                "session_id": self.session_id,
                "timestamp": detection["timestamp"],
                "object_type": detection["class_name"],
                "confidence": detection["confidence"],
                "bbox": detection["bbox"],
                "image_data": img_base64
            }

            logger.info(f"🚨 ALERT: {alert_data['object_type']} detected on camera {self.camera_id} "
                       f"model {self.model_id} with {alert_data['confidence']:.1%} confidence")

            # Broadcast alert to session-specific WebSocket clients
            asyncio.run_coroutine_threadsafe(
                self._broadcast_session_alert(camera_manager, alert_data),
                camera_manager.loop
            )
            
        except Exception as e:
            logger.error(f"Error sending alert for session {self.session_id}: {e}")
            
    async def _broadcast_session_alert(self, camera_manager, alert_data):
        """Broadcast alert to WebSocket clients for this session"""
        alert_message = {
            "type": "alert",
            "camera_id": self.camera_id,
            "model": self.model_id,
            "session_id": self.session_id,
            "alert": alert_data
        }
        
        disconnected_clients = []
        
        # Send to session-specific clients
        for client in self.websocket_clients:
            try:
                await client.send_json(alert_message)
            except Exception as e:
                logger.error(f"Failed to send alert to session client: {e}")
                disconnected_clients.append(client)
        
        # Send to general camera clients (for backward compatibility)
        str_camera_id = str(self.camera_id)
        if str_camera_id in camera_manager.connected_clients:
            for client in camera_manager.connected_clients[str_camera_id]:
                try:
                    await client.send_json(alert_message)
                except Exception as e:
                    logger.error(f"Failed to send alert to general client: {e}")
                    disconnected_clients.append(client)
        
        # Clean up disconnected clients
        for client in disconnected_clients:
            if client in self.websocket_clients:
                self.websocket_clients.remove(client)

class MultiModelManager:
    """Manages multiple detection models and sessions"""
    def __init__(self):
        self.loaded_models: Dict[str, YOLO] = {}
        self.detection_sessions: Dict[str, DetectionSession] = {}  # session_id -> session
        self.camera_sessions: Dict[str, List[str]] = {}  # camera_id -> [session_ids]
        
    def load_model(self, model_id: str, model_path: str) -> YOLO:
        """Load a model if not already loaded"""
        if model_id not in self.loaded_models:
            logger.info(f"Loading model {model_id} from {model_path}")
            self.loaded_models[model_id] = YOLO(model_path)
        return self.loaded_models[model_id]
        
    def create_session(self, camera_id: str, model_id: str, model_path: str, 
                      session_id: str = None) -> DetectionSession:
        """Create a new detection session"""
        if not session_id:
            session_id = f"{camera_id}_{model_id}_{int(time.time())}"
            
        # Load model if not already loaded
        model = self.load_model(model_id, model_path)
        
        # Create session
        session = DetectionSession(session_id, camera_id, model_id, model)
        self.detection_sessions[session_id] = session
        
        # Track sessions per camera
        if camera_id not in self.camera_sessions:
            self.camera_sessions[camera_id] = []
        self.camera_sessions[camera_id].append(session_id)
        
        logger.info(f"Created detection session {session_id} for camera {camera_id} with model {model_id}")
        return session
        
    def remove_session(self, session_id: str):
        """Remove a detection session"""
        if session_id in self.detection_sessions:
            session = self.detection_sessions[session_id]
            session.stop_processing()
            
            # Remove from camera sessions
            if session.camera_id in self.camera_sessions:
                if session_id in self.camera_sessions[session.camera_id]:
                    self.camera_sessions[session.camera_id].remove(session_id)
                    
                # Clean up empty camera entries
                if not self.camera_sessions[session.camera_id]:
                    del self.camera_sessions[session.camera_id]
            
            del self.detection_sessions[session_id]
            logger.info(f"Removed detection session {session_id}")
            
    def get_camera_sessions(self, camera_id: str) -> List[DetectionSession]:
        """Get all active sessions for a camera"""
        session_ids = self.camera_sessions.get(camera_id, [])
        return [self.detection_sessions[sid] for sid in session_ids 
                if sid in self.detection_sessions]
                
    def get_session(self, session_id: str) -> Optional[DetectionSession]:
        """Get a specific session"""
        return self.detection_sessions.get(session_id)
        
    def cleanup(self):
        """Clean up all sessions and models"""
        logger.info("Cleaning up multi-model manager...")
        
        # Stop all sessions
        for session in self.detection_sessions.values():
            session.stop_processing()
            
        self.detection_sessions.clear()
        self.camera_sessions.clear()
        self.loaded_models.clear()

class CameraThread:
    """Enhanced camera processing thread with multi-model support"""
    def __init__(self, camera_id, camera, camera_manager):
        self.camera_id = camera_id
        self.camera = camera
        self.camera_manager = camera_manager
        self.active = True
        self.cap = None
        self.thread = None
        self.frame_queue = queue.Queue(maxsize=MAX_FRAME_QUEUE_SIZE)
        self.last_frame_time = 0
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.reconnect_delay = 2
        
    def start(self):
        """Start the camera processing thread"""
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        logger.info(f"Started camera thread for {self.camera_id}")
        
    def stop(self):
        """Stop the camera processing thread"""
        self.active = False
        if self.cap:
            self.cap.release()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)
        logger.info(f"Stopped camera thread for {self.camera_id}")
        
    def _initialize_camera(self):
        """Initialize camera with proper error handling"""
        try:
            source = self._get_camera_source()
            self.cap = cv2.VideoCapture(source)
            
            # Set buffer size to reduce latency
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Configure camera properties
            if self.camera.resolution:
                width, height = map(int, self.camera.resolution.split('x'))
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
            if self.camera.fps:
                self.cap.set(cv2.CAP_PROP_FPS, self.camera.fps)
            
            # Test if camera is working
            if not self.cap.isOpened():
                raise Exception(f"Failed to open camera source: {source}")
                
            # Test read a frame
            ret, frame = self.cap.read()
            if not ret or frame is None:
                raise Exception(f"Failed to read frame from camera: {source}")
                
            self.camera_manager.camera_status[self.camera_id] = "connected"
            self.reconnect_attempts = 0
            logger.info(f"Camera {self.camera_id} initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize camera {self.camera_id}: {e}")
            self.camera_manager.camera_status[self.camera_id] = "error"
            if self.cap:
                self.cap.release()
                self.cap = None
            return False
    
    def _get_camera_source(self):
        """Convert the camera source to the proper format for OpenCV"""
        if self.camera.type == "webcam":
            return int(self.camera.source)
        else:
            return self.camera.source
    
    def _reconnect(self):
        """Attempt to reconnect to camera"""
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            logger.error(f"Max reconnection attempts reached for camera {self.camera_id}")
            self.camera_manager.camera_status[self.camera_id] = "error"
            return False
            
        self.reconnect_attempts += 1
        logger.info(f"Attempting to reconnect camera {self.camera_id} (attempt {self.reconnect_attempts})")
        
        if self.cap:
            self.cap.release()
            
        time.sleep(self.reconnect_delay)
        return self._initialize_camera()
    
    def _run(self):
        """Main camera processing loop with multi-model support"""
        if not self._initialize_camera():
            return
            
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        frame_interval = 1.0 / fps
        pre_recording_frames_count = int(fps * PRE_RECORDING_BUFFER_SECONDS)
        
        recording_writer = None
        recording_path = None
        
        while self.active:
            try:
                current_time = time.time()
                
                # Rate limiting
                if current_time - self.last_frame_time < frame_interval:
                    time.sleep(0.001)
                    continue
                    
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    logger.warning(f"Failed to read frame from camera {self.camera_id}")
                    self.camera_manager.camera_status[self.camera_id] = "disconnected"
                    
                    if not self._reconnect():
                        break
                    continue
                
                # Update status and timestamp
                self.camera_manager.camera_status[self.camera_id] = "connected"
                self.last_frame_time = current_time
                
                # Store latest frame (non-blocking)
                self.camera_manager.camera_frames[self.camera_id] = frame.copy()
                
                # Update pre-recording buffer
                pre_buffer = self.camera_manager.pre_recording_buffers[self.camera_id]
                pre_buffer.append(frame.copy())
                if len(pre_buffer) > pre_recording_frames_count:
                    pre_buffer.pop(0)
                
                # Queue frame for ALL active detection sessions for this camera
                if self.camera.detection_enabled:
                    sessions = self.camera_manager.multi_model_manager.get_camera_sessions(self.camera_id)
                    for session in sessions:
                        if session.active and not session.detection_queue.full():
                            try:
                                session.detection_queue.put_nowait((frame.copy(), current_time))
                            except queue.Full:
                                pass  # Skip if queue is full
                
                # Handle recording (unchanged)
                if self.camera.recording:
                    if recording_writer is None:
                        recording_writer, recording_path = self._start_recording()
                    
                    if recording_writer:
                        if recording_path and len(pre_buffer) > 0:
                            for buffer_frame in pre_buffer:
                                recording_writer.write(buffer_frame)
                            pre_buffer.clear()
                        
                        recording_writer.write(frame)
                        
                elif recording_writer is not None:
                    recording_writer.release()
                    recording_writer = None
                    recording_path = None
                
            except Exception as e:
                logger.error(f"Error in camera thread {self.camera_id}: {e}")
                if not self._reconnect():
                    break
        
        # Cleanup
        if recording_writer:
            recording_writer.release()
        if self.cap:
            self.cap.release()
        self.camera_manager.camera_status[self.camera_id] = "disconnected"
        logger.info(f"Camera thread {self.camera_id} stopped")
    
    def _start_recording(self):
        """Start recording with proper error handling"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            recording_path = f"{RECORDINGS_DIR}{self.camera_id}_{timestamp}.mp4"
            
            # Get video properties
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            recording_writer = cv2.VideoWriter(recording_path, fourcc, fps, (width, height))
            
            if recording_writer.isOpened():
                logger.info(f"Started recording for camera {self.camera_id}: {recording_path}")
                return recording_writer, recording_path
            else:
                logger.error(f"Failed to start recording for camera {self.camera_id}")
                return None, None
                
        except Exception as e:
            logger.error(f"Error starting recording for camera {self.camera_id}: {e}")
            return None, None

class CameraManager:
    """Enhanced camera manager with multi-model support"""
    def __init__(self):
        self.cameras: Dict[Union[int, str], Camera] = {}
        self.camera_threads: Dict[Union[int, str], CameraThread] = {}
        self.camera_frames: Dict[Union[int, str], np.ndarray] = {}
        self.camera_status: Dict[Union[int, str], str] = {}
        self.pre_recording_buffers: Dict[Union[int, str], List] = {}
        self.active = True
        self.connected_clients: Dict[str, List[WebSocket]] = {}
        self.loop = None
        self.recording_config = RecordingConfig()
        self.active_recordings: Dict[str, RecordingInfo] = {}
        self.recording_history: List[RecordingInfo] = []
        # Multi-model manager
        self.multi_model_manager = MultiModelManager()
        
        # Load available models
        self.available_models = self._load_available_models()
        
        # Load camera configurations
        self._load_camera_config()
    
    def set_event_loop(self, loop):
        """Set the event loop for async operations"""
        self.loop = loop
    
    def _load_available_models(self) -> Dict[str, Any]:
        models = {}
        default_models = [
            {"id": "yolov8n", "name": "YOLOv8 Nano", "path": "./models/yolov8n.pt"},
            {"id": "yolov8s", "name": "YOLOv8 Small", "path": "./models/yolov8s.pt"},
            {"id": "yolov8m", "name": "YOLOv8 Medium", "path": "./models/yolov8m.pt"},
            {"id": "yolov8l", "name": "YOLOv8 Large", "path": "./models/yolov8l.pt"},
        ]
        
        for model in default_models:
            models[model["id"]] = model
            
        return models
    
    def _load_camera_config(self):
        try:
            with open(CAMERA_CONFIG_PATH, 'r') as f:
                config_data = json.load(f)
                
            if isinstance(config_data, dict) and "cameras" in config_data:
                camera_configs = config_data["cameras"]
            else:
                camera_configs = config_data
                
            for camera_config in camera_configs:
                camera_id = camera_config.get("id")
                self.cameras[camera_id] = Camera(**camera_config)
                self.pre_recording_buffers[camera_id] = []
                self.camera_status[camera_id] = "disconnected"
                self.connected_clients[str(camera_id)] = []
                
        except FileNotFoundError:
            logger.info(f"Camera configuration file not found at {CAMERA_CONFIG_PATH}")
            with open(CAMERA_CONFIG_PATH, 'w') as f:
                json.dump({"cameras": []}, f)
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in camera configuration file")
    
    def _save_camera_config(self):
        with open(CAMERA_CONFIG_PATH, 'w') as f:
            config = {"cameras": [cam.model_dump() for cam in self.cameras.values()]}
            json.dump(config, f, indent=2)
    
    def start_camera_threads(self):
        """Start camera processing threads for enabled cameras"""
        for camera_id, camera in self.cameras.items():
            if camera.enabled and camera_id not in self.camera_threads:
                camera_thread = CameraThread(camera_id, camera, self)
                self.camera_threads[camera_id] = camera_thread
                camera_thread.start()
    
    async def start_tracking_session(self, camera_id: str, request: TrackingRequest):
        """Start a new tracking session for camera-model combination"""
        if camera_id not in self.cameras:
            raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")
        
        model_id = request.model
        if model_id not in self.available_models:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
        
        # Generate session ID
        session_id = request.session_id or request.pair_id or f"{camera_id}_{model_id}_{int(time.time())}"
        
        # Check if session already exists
        if self.multi_model_manager.get_session(session_id):
            logger.info(f"Session {session_id} already exists, reusing")
            return {"session_id": session_id, "message": f"Session already active"}
        
        # Create new detection session
        model_info = self.available_models[model_id]
        session = self.multi_model_manager.create_session(
            camera_id, model_id, model_info["path"], session_id
        )
        
        # Start processing for this session
        session.start_processing(self)
        
        # Enable detection for camera
        self.cameras[camera_id].detection_enabled = True
        
        logger.info(f"Started tracking session {session_id} for camera {camera_id} with model {model_id}")
        
        return {
            "session_id": session_id,
            "camera_id": camera_id,
            "model": model_id,
            "message": f"Tracking started for camera {camera_id} with model {model_id}"
        }
    
    async def stop_tracking_session(self, camera_id: str, model_id: str = None, session_id: str = None):
        """Stop a specific tracking session"""
        if camera_id not in self.cameras:
            raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")
        
        if session_id:
            # Stop specific session
            self.multi_model_manager.remove_session(session_id)
            message = f"Stopped tracking session {session_id}"
        elif model_id:
            # Stop all sessions for this camera-model combination
            sessions = self.multi_model_manager.get_camera_sessions(camera_id)
            removed_sessions = []
            for session in sessions:
                if session.model_id == model_id:
                    self.multi_model_manager.remove_session(session.session_id)
                    removed_sessions.append(session.session_id)
            message = f"Stopped {len(removed_sessions)} sessions for camera {camera_id} model {model_id}"
        else:
            # Stop all sessions for this camera
            sessions = self.multi_model_manager.get_camera_sessions(camera_id)
            for session in sessions:
                self.multi_model_manager.remove_session(session.session_id)
            message = f"Stopped all tracking for camera {camera_id}"
        
        # Disable detection if no more sessions
        remaining_sessions = self.multi_model_manager.get_camera_sessions(camera_id)
        if not remaining_sessions:
            self.cameras[camera_id].detection_enabled = False
        
        return {"message": message}
    
    def get_camera_detection_status(self, camera_id: str, model_id: str = None):
        """Get detection status for camera, optionally filtered by model"""
        if camera_id not in self.cameras:
            raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")
        
        camera = self.cameras[camera_id]
        sessions = self.multi_model_manager.get_camera_sessions(camera_id)
        
        if model_id:
            sessions = [s for s in sessions if s.model_id == model_id]
        
        # Aggregate detections from all sessions
        all_detections = []
        active_models = []
        
        for session in sessions:
            all_detections.extend(session.detections)
            active_models.append(session.model_id)
        
        # Get recent detections (last minute)
        recent_detections = [d for d in all_detections if 
                           (datetime.now() - datetime.fromisoformat(d["timestamp"])).seconds < 60]
        
        return {
            "camera_id": camera_id,
            "detection_enabled": camera.detection_enabled,
            "camera_status": self.camera_status.get(camera_id, "disconnected"),
            "active_sessions": len(sessions),
            "active_models": active_models,
            "total_detections": len(all_detections),
            "recent_detections": len(recent_detections),
            "last_detection": all_detections[0] if all_detections else None,
            "session_details": [
                {
                    "session_id": s.session_id,
                    "model": s.model_id,
                    "detections": len(s.detections),
                    "websocket_clients": len(s.websocket_clients)
                }
                for s in sessions
            ]
        }
    
    def get_multi_model_stream_generator(self, camera_id: str, model_id: str = None, session_id: str = None):
        """Get frame generator with model-specific detections"""
        if camera_id not in self.cameras:
            raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")
        
        def generate():
            while self.active:
                if camera_id in self.camera_frames:
                    frame = self.camera_frames[camera_id].copy()
                    
                    # Get sessions to display
                    if session_id:
                        session = self.multi_model_manager.get_session(session_id)
                        sessions = [session] if session else []
                    elif model_id:
                        all_sessions = self.multi_model_manager.get_camera_sessions(camera_id)
                        sessions = [s for s in all_sessions if s.model_id == model_id]
                    else:
                        sessions = self.multi_model_manager.get_camera_sessions(camera_id)
                    
                    # Draw detections from selected sessions
                    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
                    
                    for i, session in enumerate(sessions):
                        color = colors[i % len(colors)]
                        for det in session.detections:
                            x1, y1, x2, y2 = map(int, det["bbox"])
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            
                            # Add model label to distinguish between models
                            label = f"{det['class_name']} {det['confidence']:.2f} [{session.model_id}]"
                            cv2.putText(frame, label, (x1, y1 - 10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # Encode frame as JPEG
                    _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
                else:
                    # Return black frame if camera not available
                    black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    _, jpeg = cv2.imencode('.jpg', black_frame)
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
                
                time.sleep(0.033)  # ~30 FPS
        
        return generate
    def start_recording(self, camera_id: str, output_dir: str = None) -> RecordingInfo:
        """Start recording for a specific camera"""
        if camera_id not in self.cameras:
            raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")
        
        if camera_id in self.active_recordings:
            raise HTTPException(status_code=400, detail=f"Camera {camera_id} is already recording")
        
        # Use provided output dir or default
        output_directory = output_dir or self.recording_config.output_directory or RECORDINGS_DIR
        
        # Ensure output directory exists
        os.makedirs(output_directory, exist_ok=True)
        
        # Generate recording info
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"camera_{camera_id}_{timestamp}.mp4"
        file_path = os.path.join(output_directory, filename)
        
        recording_info = RecordingInfo(
            id=f"{camera_id}_{timestamp}",
            camera_id=camera_id,
            filename=filename,
            file_path=file_path,
            start_time=datetime.now(),
            status="recording"
        )
        
        # Start recording on camera
        self.cameras[camera_id].recording = True
        self.active_recordings[camera_id] = recording_info
        
        logger.info(f"Started recording for camera {camera_id}: {file_path}")
        return recording_info
    
    def stop_recording(self, camera_id: str) -> RecordingInfo:
        """Stop recording for a specific camera"""
        if camera_id not in self.cameras:
            raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")
        
        if camera_id not in self.active_recordings:
            raise HTTPException(status_code=400, detail=f"Camera {camera_id} is not recording")
        
        # Stop recording on camera
        self.cameras[camera_id].recording = False
        
        # Update recording info
        recording_info = self.active_recordings[camera_id]
        recording_info.end_time = datetime.now()
        recording_info.duration_seconds = (recording_info.end_time - recording_info.start_time).total_seconds()
        recording_info.status = "completed"
        
        # Get file size if file exists
        if os.path.exists(recording_info.file_path):
            recording_info.file_size_bytes = os.path.getsize(recording_info.file_path)
        
        # Move to history and remove from active
        self.recording_history.append(recording_info)
        del self.active_recordings[camera_id]
        
        logger.info(f"Stopped recording for camera {camera_id}: {recording_info.file_path}")
        return recording_info
    
    def get_recording_status(self, camera_id: str) -> bool:
        """Check if camera is currently recording"""
        return camera_id in self.active_recordings
    
    def get_recordings(self, camera_id: str = None, date: str = None) -> List[RecordingInfo]:
        """Get recording history with optional filtering"""
        recordings = self.recording_history.copy()
        
        # Add currently active recordings
        for active_recording in self.active_recordings.values():
            recordings.append(active_recording)
        
        # Filter by camera if specified
        if camera_id:
            recordings = [r for r in recordings if r.camera_id == camera_id]
        
        # Filter by date if specified (YYYY-MM-DD format)
        if date:
            try:
                filter_date = datetime.strptime(date, "%Y-%m-%d").date()
                recordings = [r for r in recordings if r.start_time.date() == filter_date]
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
        
        # Sort by start time (newest first)
        recordings.sort(key=lambda x: x.start_time, reverse=True)
        
        return recordings
    
    def delete_recording(self, recording_id: str) -> bool:
        """Delete a recording file and remove from history"""
        # Find recording in history
        recording_to_delete = None
        for i, recording in enumerate(self.recording_history):
            if recording.id == recording_id:
                recording_to_delete = recording
                break
        
        if not recording_to_delete:
            raise HTTPException(status_code=404, detail=f"Recording {recording_id} not found")
        
        # Delete file if it exists
        if os.path.exists(recording_to_delete.file_path):
            try:
                os.remove(recording_to_delete.file_path)
                logger.info(f"Deleted recording file: {recording_to_delete.file_path}")
            except Exception as e:
                logger.error(f"Failed to delete recording file: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to delete recording file: {e}")
        
        # Remove from history
        self.recording_history = [r for r in self.recording_history if r.id != recording_id]
        
        return True
    
    def update_recording_config(self, config: RecordingConfig):
        """Update recording configuration"""
        self.recording_config = config
        
        # Create output directory if specified
        if config.output_directory:
            os.makedirs(config.output_directory, exist_ok=True)
    # ... existing methods remain the same ...
    def get_cameras(self):
        """Get all configured cameras with their status"""
        camera_list = []
        for camera_id, camera in self.cameras.items():
            camera_dict = camera.model_dump()
            camera_dict["status"] = self.camera_status.get(camera_id, "disconnected")
            camera_list.append(camera_dict)
        return camera_list
    
    def cleanup(self):
        """Clean up resources before shutting down"""
        logger.info("Starting cleanup...")
        self.active = False
        
        # Stop all camera threads
        for camera_id, camera_thread in self.camera_threads.items():
            camera_thread.stop()
        
        # Clean up multi-model manager
        self.multi_model_manager.cleanup()
        
        # Clear all data
        self.camera_threads.clear()
        self.camera_frames.clear()
        
        logger.info("Cleanup completed")

# Initialize camera manager (will be set in lifespan)
camera_manager = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    global camera_manager
    camera_manager = CameraManager()
    
    # Set event loop for camera manager
    loop = asyncio.get_event_loop()
    camera_manager.set_event_loop(loop)
    
    # Start camera threads
    camera_manager.start_camera_threads()
    
    yield
    
    # Shutdown logic
    if camera_manager:
        camera_manager.cleanup()

app = FastAPI(title="Multi-Camera YOLO Detection API", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Enhanced API routes for multi-model support

@app.get("/api/cameras", response_model=List[Dict])
async def get_cameras():
    """Get all configured cameras"""
    return camera_manager.get_cameras()

@app.post("/api/cameras/{camera_id}/start_tracking")
async def start_tracking(camera_id: Union[int, str], request: TrackingRequest):
    """Start tracking for a camera with specific model (supports multi-model)"""
    return await camera_manager.start_tracking_session(str(camera_id), request)

@app.post("/api/cameras/{camera_id}/start_tracking_multimodel") 
async def start_tracking_multimodel(camera_id: Union[int, str], request: TrackingRequest):
    """Enhanced endpoint for multi-model tracking"""
    return await camera_manager.start_tracking_session(str(camera_id), request)

@app.post("/api/cameras/{camera_id}/stop_tracking")
async def stop_tracking(camera_id: Union[int, str], request: dict = None):
    """Stop tracking for a camera (supports model-specific stopping)"""
    model_id = request.get("model") if request else None
    session_id = request.get("pair_id") if request else None
    
    return await camera_manager.stop_tracking_session(str(camera_id), model_id, session_id)

@app.post("/api/cameras/{camera_id}/stop_tracking_model")
async def stop_tracking_model(camera_id: Union[int, str], request: dict):
    """Stop tracking for specific model on a camera"""
    model_id = request.get("model")
    session_id = request.get("pair_id")
    
    if not model_id and not session_id:
        raise HTTPException(status_code=400, detail="Either model or pair_id must be provided")
    
    return await camera_manager.stop_tracking_session(str(camera_id), model_id, session_id)

@app.get("/api/cameras/{camera_id}/detection_stream")
async def stream_detection_video(
    camera_id: Union[int, str], 
    model: str = None, 
    pair_id: str = None,
    session: str = None
):
    """Stream processed detection frames with model-specific bounding boxes"""
    return StreamingResponse(
        camera_manager.get_multi_model_stream_generator(
            str(camera_id), model, pair_id or session
        )(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache", 
            "Expires": "0"
        }
    )

@app.get("/api/cameras/{camera_id}/stream")
async def stream_camera(
    camera_id: Union[int, str],
    pair_id: str = None,
    session: str = None
):
    """Stream regular video from a camera"""
    if camera_id not in camera_manager.cameras:
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")
    
    def generate():
        while camera_manager.active:
            if camera_id in camera_manager.camera_frames:
                frame = camera_manager.camera_frames[camera_id].copy()
                _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            else:
                black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                _, jpeg = cv2.imencode('.jpg', black_frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            
            time.sleep(0.033)  # ~30 FPS

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/api/cameras/{camera_id}/detection_status")
async def get_detection_status(camera_id: Union[int, str], model: str = None):
    """Get real-time detection status for a camera with multi-model support"""
    return camera_manager.get_camera_detection_status(str(camera_id), model)

@app.get("/api/cameras/{camera_id}/multimodel_status")
async def get_multimodel_status(camera_id: Union[int, str]):
    """Get multi-model status for a camera"""
    sessions = camera_manager.multi_model_manager.get_camera_sessions(str(camera_id))
    
    return {
        "camera_id": camera_id,
        "total_sessions": len(sessions),
        "models": [s.model_id for s in sessions],
        "active_models": list(set(s.model_id for s in sessions if s.active)),
        "sessions": [
            {
                "session_id": s.session_id,
                "model": s.model_id,
                "active": s.active,
                "detections": len(s.detections),
                "websocket_clients": len(s.websocket_clients)
            }
            for s in sessions
        ]
    }

@app.get("/api/capabilities/multimodel")
async def get_multimodel_capabilities():
    """Check if multi-model support is available"""
    return {
        "supported": True,
        "max_models_per_camera": 4,
        "available_models": list(camera_manager.available_models.keys())
    }

@app.websocket("/ws/cameras/{camera_id}")
async def websocket_camera_alerts(websocket: WebSocket, camera_id: Union[int, str]):
    """WebSocket endpoint for camera alerts (backward compatibility)"""
    await websocket.accept()
    str_camera_id = str(camera_id)
    
    await websocket.send_json({
        "type": "connection",
        "status": "connected", 
        "camera_id": str_camera_id,
        "message": f"Connected to alerts for camera {camera_id}",
        "timestamp": datetime.now().isoformat()
    })
    
    # Register this client for general camera alerts
    if str_camera_id not in camera_manager.connected_clients:
        camera_manager.connected_clients[str_camera_id] = []
    camera_manager.connected_clients[str_camera_id].append(websocket)
    
    try:
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                message = json.loads(data)
                message_type = message.get("type")
                
                if message_type == "ping":
                    await websocket.send_json({
                        "type": "pong",
                        "timestamp": datetime.now().isoformat()
                    })
                elif message_type == "get_status":
                    status = camera_manager.get_camera_detection_status(str_camera_id)
                    await websocket.send_json({
                        "type": "status",
                        "data": status
                    })
                        
            except asyncio.TimeoutError:
                await websocket.send_json({
                    "type": "heartbeat",
                    "timestamp": datetime.now().isoformat()
                })
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected from camera {camera_id}")
    except Exception as e:
        logger.error(f"WebSocket error for camera {camera_id}: {e}")
    finally:
        if websocket in camera_manager.connected_clients.get(str_camera_id, []):
            camera_manager.connected_clients[str_camera_id].remove(websocket)

@app.websocket("/ws/cameras/{camera_id}/{model}")
async def websocket_model_specific_alerts(
    websocket: WebSocket, 
    camera_id: Union[int, str], 
    model: str
):
    """WebSocket endpoint for model-specific alerts"""
    await websocket.accept()
    str_camera_id = str(camera_id)
    
    await websocket.send_json({
        "type": "connection",
        "status": "connected",
        "camera_id": str_camera_id,
        "model": model,
        "message": f"Connected to alerts for camera {camera_id} model {model}",
        "timestamp": datetime.now().isoformat()
    })
    
    # Find sessions for this camera-model combination
    sessions = camera_manager.multi_model_manager.get_camera_sessions(str_camera_id)
    model_sessions = [s for s in sessions if s.model_id == model]
    
    # Register this client with all matching sessions
    for session in model_sessions:
        session.websocket_clients.append(websocket)
    
    try:
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                message = json.loads(data)
                message_type = message.get("type")
                
                if message_type == "ping":
                    await websocket.send_json({
                        "type": "pong",
                        "timestamp": datetime.now().isoformat()
                    })
                elif message_type == "get_status":
                    status = camera_manager.get_camera_detection_status(str_camera_id, model)
                    await websocket.send_json({
                        "type": "status",
                        "data": status
                    })
                        
            except asyncio.TimeoutError:
                await websocket.send_json({
                    "type": "heartbeat",
                    "timestamp": datetime.now().isoformat()
                })
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected from camera {camera_id} model {model}")
    except Exception as e:
        logger.error(f"WebSocket error for camera {camera_id} model {model}: {e}")
    finally:
        # Remove client from all sessions
        for session in model_sessions:
            if websocket in session.websocket_clients:
                session.websocket_clients.remove(websocket)

# @app.get("/api/health")
# async def health_check():
#     return {
#         "status": "ok",
#         "active_cameras": len([s for s in camera_manager.camera_status.values() if s == "connected"]),
#         "total_cameras": len(camera_manager.cameras),
#         "multi_model_support": True,
#         "active_sessions": len(camera_manager.multi_model_manager.detection_sessions),
#         "loaded_models": len(camera_manager.multi_model_manager.loaded_models)
#     }


@app.post("/api/cameras/{camera_id}/start_recording")
async def start_recording(
    camera_id: Union[int, str], 
    request: dict = None
):
    """Start recording for a specific camera"""
    output_dir = request.get("output_dir") if request else None
    
    try:
        recording_info = camera_manager.start_recording(str(camera_id), output_dir)
        return {
            "success": True,
            "message": f"Recording started for camera {camera_id}",
            "recording": recording_info.model_dump()
        }
    except Exception as e:
        logger.error(f"Failed to start recording for camera {camera_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/cameras/{camera_id}/stop_recording")
async def stop_recording(camera_id: Union[int, str]):
    """Stop recording for a specific camera"""
    try:
        recording_info = camera_manager.stop_recording(str(camera_id))
        return {
            "success": True,
            "message": f"Recording stopped for camera {camera_id}",
            "recording": recording_info.model_dump()
        }
    except Exception as e:
        logger.error(f"Failed to stop recording for camera {camera_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/cameras/{camera_id}/recording_status")
async def get_recording_status(camera_id: Union[int, str]):
    """Check if camera is currently recording"""
    try:
        is_recording = camera_manager.get_recording_status(str(camera_id))
        active_recording = camera_manager.active_recordings.get(str(camera_id))
        
        return {
            "camera_id": camera_id,
            "is_recording": is_recording,
            "recording_info": active_recording.model_dump() if active_recording else None
        }
    except Exception as e:
        logger.error(f"Failed to get recording status for camera {camera_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/recordings")
async def get_recordings(
    camera_id: Union[int, str] = None,
    date: str = None,
    limit: int = 50
):
    """Get recording history with optional filtering"""
    try:
        recordings = camera_manager.get_recordings(
            camera_id=str(camera_id) if camera_id else None,
            date=date
        )
        
        # Limit results
        recordings = recordings[:limit]
        
        return {
            "recordings": [r.model_dump() for r in recordings],
            "total": len(recordings),
            "active_recordings": len(camera_manager.active_recordings)
        }
    except Exception as e:
        logger.error(f"Failed to get recordings: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/recordings/{recording_id}")
async def delete_recording(recording_id: str):
    """Delete a specific recording"""
    try:
        success = camera_manager.delete_recording(recording_id)
        return {
            "success": success,
            "message": f"Recording {recording_id} deleted successfully"
        }
    except Exception as e:
        logger.error(f"Failed to delete recording {recording_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/recordings/{recording_id}/download")
async def download_recording(recording_id: str):
    """Download a specific recording file"""
    # Find recording in history
    recording = None
    for r in camera_manager.recording_history:
        if r.id == recording_id:
            recording = r
            break
    
    if not recording:
        raise HTTPException(status_code=404, detail=f"Recording {recording_id} not found")
    
    if not os.path.exists(recording.file_path):
        raise HTTPException(status_code=404, detail=f"Recording file not found: {recording.filename}")
    
    def iterfile(file_path: str):
        with open(file_path, mode="rb") as file_like:
            yield from file_like
    
    return StreamingResponse(
        iterfile(recording.file_path),
        media_type="video/mp4",
        headers={"Content-Disposition": f"attachment; filename={recording.filename}"}
    )

@app.post("/api/recording/config")
async def update_recording_config(config: RecordingConfig):
    """Update recording configuration"""
    try:
        camera_manager.update_recording_config(config)
        return {
            "success": True,
            "message": "Recording configuration updated",
            "config": config.model_dump()
        }
    except Exception as e:
        logger.error(f"Failed to update recording config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/recording/config")
async def get_recording_config():
    """Get current recording configuration"""
    return camera_manager.recording_config.model_dump()

@app.get("/api/recording/status")
async def get_global_recording_status():
    """Get global recording status"""
    active_recordings = list(camera_manager.active_recordings.values())
    
    return {
        "total_cameras": len(camera_manager.cameras),
        "cameras_recording": len(active_recordings),
        "active_recordings": [r.model_dump() for r in active_recordings],
        "total_recordings_in_history": len(camera_manager.recording_history),
        "output_directory": camera_manager.recording_config.output_directory or RECORDINGS_DIR
    }

@app.post("/api/recording/start_all")
async def start_all_recordings(request: dict = None):
    """Start recording on all available cameras"""
    output_dir = request.get("output_dir") if request else None
    
    started_recordings = []
    failed_cameras = []
    
    for camera_id in camera_manager.cameras.keys():
        try:
            if not camera_manager.get_recording_status(camera_id):
                recording_info = camera_manager.start_recording(camera_id, output_dir)
                started_recordings.append(recording_info.model_dump())
        except Exception as e:
            logger.error(f"Failed to start recording for camera {camera_id}: {e}")
            failed_cameras.append({"camera_id": camera_id, "error": str(e)})
    
    return {
        "success": True,
        "message": f"Started recording on {len(started_recordings)} cameras",
        "started_recordings": started_recordings,
        "failed_cameras": failed_cameras
    }

@app.post("/api/recording/stop_all")
async def stop_all_recordings():
    """Stop recording on all cameras"""
    stopped_recordings = []
    failed_cameras = []
    
    active_camera_ids = list(camera_manager.active_recordings.keys())
    
    for camera_id in active_camera_ids:
        try:
            recording_info = camera_manager.stop_recording(camera_id)
            stopped_recordings.append(recording_info.model_dump())
        except Exception as e:
            logger.error(f"Failed to stop recording for camera {camera_id}: {e}")
            failed_cameras.append({"camera_id": camera_id, "error": str(e)})
    
    return {
        "success": True,
        "message": f"Stopped recording on {len(stopped_recordings)} cameras",
        "stopped_recordings": stopped_recordings,
        "failed_cameras": failed_cameras
    }

# Add recording info to the health check
@app.get("/api/health")
async def health_check():
    return {
        "status": "ok",
        "active_cameras": len([s for s in camera_manager.camera_status.values() if s == "connected"]),
        "total_cameras": len(camera_manager.cameras),
        "multi_model_support": True,
        "active_sessions": len(camera_manager.multi_model_manager.detection_sessions),
        "loaded_models": len(camera_manager.multi_model_manager.loaded_models),
        "recording_support": True,
        "cameras_recording": len(camera_manager.active_recordings),
        "total_recordings": len(camera_manager.recording_history)
    }


@app.get("/api/detection/health")
async def detection_health():
    """Get overall detection system health with multi-model info"""
    active_cameras = sum(1 for status in camera_manager.camera_status.values() 
                        if status == "connected")
    
    detection_enabled_cameras = sum(1 for cam in camera_manager.cameras.values() 
                                   if cam.detection_enabled)
    
    total_connections = sum(len(clients) for clients in camera_manager.connected_clients.values())
    
    # Add session-specific connections
    total_session_connections = sum(len(session.websocket_clients) 
                                   for session in camera_manager.multi_model_manager.detection_sessions.values())
    
    return {
        "multi_model_support": True,
        "loaded_models": list(camera_manager.multi_model_manager.loaded_models.keys()),
        "total_cameras": len(camera_manager.cameras),
        "active_cameras": active_cameras,
        "detection_enabled_cameras": detection_enabled_cameras,
        "active_sessions": len(camera_manager.multi_model_manager.detection_sessions),
        "total_websocket_connections": total_connections + total_session_connections,
        "system_status": "healthy"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)