# Enhanced main.py with Database Integration
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks, Response, APIRouter, Depends
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List, Dict, Optional, Any, Union
import json
import cv2
import numpy as np
import asyncio
import threading
import time
import os
import uuid
from datetime import datetime, timedelta
import torch
from ultralytics import YOLO
from contextlib import asynccontextmanager
import base64
import random
import queue
import logging
from concurrent.futures import ThreadPoolExecutor
import weakref
import psutil

# Database imports
from database import get_db, init_database, test_connection, get_db_session
from models import (
    CameraModel, ModelInfoModel, DetectionSessionModel, DetectionModel,
    AlertModel, RecordingModel, SystemStatsModel, UserActivityModel
)
from schemas import (
    Camera, CameraCreate, CameraUpdate,
    ModelInfo, ModelInfoCreate, ModelInfoUpdate,
    DetectionSession, DetectionSessionCreate, DetectionSessionUpdate,
    Detection, DetectionCreate, DetectionFilter,
    Alert, AlertCreate, AlertUpdate, AlertFilter,
    RecordingInfo, RecordingInfoCreate, RecordingInfoUpdate, RecordingConfig, RecordingFilter,
    TrackingRequest, TrackingResponse, DetectionStatusResponse, MultiModelStatusResponse,
    SystemStats, SystemStatsCreate, UserActivityCreate, UserActivity,
    PaginationParams, PaginatedResponse,
    HealthResponse, DetectionHealthResponse
)
from services import (
    CameraService, ModelService, SessionService, DetectionService,
    AlertService, RecordingService, SystemStatsService, UserActivityService,
    DatabaseMaintenanceService
)

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
MAX_WORKERS = 8

# Ensure directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RECORDINGS_DIR, exist_ok=True)

class DetectionSession:
    """Enhanced detection session with database integration"""
    def __init__(self, session_id: str, camera_id: str, model_id: str, model: YOLO, db_session_id: str):
        self.session_id = session_id
        self.camera_id = camera_id
        self.model_id = model_id
        self.model = model
        self.db_session_id = db_session_id  # Store session ID instead of object
        self.active = True
        self.detection_queue = queue.Queue(maxsize=5)
        self.detections = []
        self.processing_thread = None
        self.websocket_clients = []
        self.error_count = 0  # Track errors locally
        
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
        """Process detections for this specific session with database storage"""
        while self.active:
            try:
                if not self.detection_queue.empty():
                    frame, timestamp = self.detection_queue.get_nowait()
                    
                    start_time = time.time()
                    # Run detection with this session's model
                    results = self.model.predict([frame], conf=0.25, verbose=False)
                    inference_time = time.time() - start_time
                    
                    # Get camera from database
                    with get_db_session() as db:
                        db_camera = CameraService.get_camera(db, self.camera_id)
                        if not db_camera or not db_camera.detection_enabled:
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
                                    
                                    # Create detection object for API response
                                    detection_data = {
                                        "bbox": [x1, y1, x2, y2],
                                        "confidence": confidence,
                                        "class_id": class_id,
                                        "class_name": class_name,
                                        "timestamp": datetime.fromtimestamp(timestamp).isoformat(),
                                        "model": self.model_id,
                                        "session_id": self.session_id
                                    }
                                    detections.append(detection_data)
                                    
                                    # Store detection in database
                                    detection_create = DetectionCreate(
                                        session_id=self.session_id,
                                        camera_id=self.camera_id,
                                        model_id=self.model_id,
                                        bbox=[x1, y1, x2, y2],
                                        confidence=confidence,
                                        class_id=class_id,
                                        class_name=class_name,
                                        frame_width=frame.shape[1],
                                        frame_height=frame.shape[0],
                                        frame_timestamp=datetime.fromtimestamp(timestamp),
                                        inference_time=inference_time
                                        # Note: meta_data is optional and not set here
                                    )
                                    
                                    db_detection = DetectionService.create_detection(db, detection_create)
                                    
                                    # Send alert if confidence exceeds threshold
                                    if confidence > db_camera.alert_threshold:
                                        self._send_alert(camera_manager, detection_data, frame, db, db_detection.id)
                        
                        # Update session statistics
                        SessionService.update_session_stats(
                            db, self.session_id, 
                            frames_processed=1, 
                            processing_time=inference_time
                        )
                        
                        # Update model statistics
                        ModelService.update_model_stats(db, self.model_id, inference_time)
                    
                    # Update in-memory detections for backward compatibility
                    self.detections = detections
                    
                    # Update central detection storage for backward compatibility
                    if not hasattr(camera_manager, 'camera_detections'):
                        camera_manager.camera_detections = {}
                    camera_manager.camera_detections[self.camera_id] = detections
                    
                else:
                    time.sleep(0.01)
                    
            except Exception as e:
                logger.error(f"Error in detection session {self.session_id}: {e}")
                # Log error to database
                try:
                    with get_db_session() as db:
                        # Increment local error count
                        self.error_count += 1
                        
                        session_update = DetectionSessionUpdate(
                            last_error=str(e),
                            last_error_at=datetime.now(),
                            error_count=self.error_count
                        )
                        SessionService.update_session(db, self.session_id, session_update)
                except Exception as db_error:
                    logger.error(f"Failed to log error to database: {db_error}")
                time.sleep(0.1)

    def _send_alert(self, camera_manager, detection, frame, db: Session, detection_id: int):
        """Send an alert for this specific session with database storage"""
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

            # Create alert in database
            alert_create = AlertCreate(
                detection_id=detection_id,
                session_id=self.session_id,
                camera_id=self.camera_id,
                model_id=self.model_id,
                object_type=detection["class_name"],
                confidence=detection["confidence"],
                bbox=detection["bbox"],
                image_data=img_base64,
                image_size=len(img_base64)
            )
            
            db_alert = AlertService.create_alert(db, alert_create)

            alert_data = {
                "id": db_alert.id,
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
            if camera_manager.loop:
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
    """Enhanced multi-model manager with database integration"""
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
        """Create a new detection session with database storage"""
        if not session_id:
            session_id = f"{camera_id}_{model_id}_{int(time.time())}"
            
        # Load model if not already loaded
        model = self.load_model(model_id, model_path)
        
        # Create session in database and get the ID immediately
        with get_db_session() as db:
            session_create = DetectionSessionCreate(
                camera_id=camera_id,
                model_id=model_id,
                session_id=session_id
            )
            db_session = SessionService.create_session(db, session_create)
            # Get the ID while the session is still bound
            db_session_id = db_session.id
        
        # Create in-memory session with the stored ID
        session = DetectionSession(session_id, camera_id, model_id, model, db_session_id)
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
            
            # Update database
            with get_db_session() as db:
                SessionService.stop_session(db, session_id)
            
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
    """Enhanced camera processing thread with database integration"""
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
        """Initialize camera with proper error handling and database update"""
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
                
            # Update camera status in database
            with get_db_session() as db:
                CameraService.update_camera_status(db, self.camera_id, "connected")
            
            self.camera_manager.camera_status[self.camera_id] = "connected"
            self.reconnect_attempts = 0
            logger.info(f"Camera {self.camera_id} initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize camera {self.camera_id}: {e}")
            
            # Update camera status in database
            try:
                with get_db_session() as db:
                    CameraService.update_camera_status(db, self.camera_id, "error")
            except Exception as db_error:
                logger.error(f"Failed to update camera status in database: {db_error}")
            
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
            try:
                with get_db_session() as db:
                    CameraService.update_camera_status(db, self.camera_id, "error")
            except Exception as db_error:
                logger.error(f"Failed to update camera status in database: {db_error}")
            self.camera_manager.camera_status[self.camera_id] = "error"
            return False
            
        self.reconnect_attempts += 1
        logger.info(f"Attempting to reconnect camera {self.camera_id} (attempt {self.reconnect_attempts})")
        
        if self.cap:
            self.cap.release()
            
        time.sleep(self.reconnect_delay)
        return self._initialize_camera()
    
    def _run(self):
        """Main camera processing loop with database integration"""
        if not self._initialize_camera():
            return
            
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        frame_interval = 1.0 / fps
        pre_recording_frames_count = int(fps * PRE_RECORDING_BUFFER_SECONDS)
        
        recording_writer = None
        recording_path = None
        current_recording_id = None
        
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
                    
                    # Update camera status in database
                    try:
                        with get_db_session() as db:
                            CameraService.update_camera_status(db, self.camera_id, "disconnected")
                    except Exception as db_error:
                        logger.error(f"Failed to update camera status in database: {db_error}")
                    self.camera_manager.camera_status[self.camera_id] = "disconnected"
                    
                    if not self._reconnect():
                        break
                    continue
                
                # Update status and timestamp
                try:
                    with get_db_session() as db:
                        CameraService.update_camera_status(db, self.camera_id, "connected")
                except Exception as db_error:
                    logger.error(f"Failed to update camera status in database: {db_error}")
                
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
                
                # Handle recording with database integration
                if self.camera.recording:
                    if recording_writer is None:
                        recording_writer, recording_path, current_recording_id = self._start_recording()
                    
                    if recording_writer:
                        if recording_path and len(pre_buffer) > 0:
                            for buffer_frame in pre_buffer:
                                recording_writer.write(buffer_frame)
                            pre_buffer.clear()
                        
                        recording_writer.write(frame)
                        
                elif recording_writer is not None:
                    self._stop_recording(recording_writer, current_recording_id)
                    recording_writer = None
                    recording_path = None
                    current_recording_id = None
                
            except Exception as e:
                logger.error(f"Error in camera thread {self.camera_id}: {e}")
                if not self._reconnect():
                    break
        
        # Cleanup
        if recording_writer:
            self._stop_recording(recording_writer, current_recording_id)
        if self.cap:
            self.cap.release()
        
        # Update final camera status
        try:
            with get_db_session() as db:
                CameraService.update_camera_status(db, self.camera_id, "disconnected")
        except Exception as db_error:
            logger.error(f"Failed to update camera status in database: {db_error}")
        self.camera_manager.camera_status[self.camera_id] = "disconnected"
        logger.info(f"Camera thread {self.camera_id} stopped")
    
    def _start_recording(self):
        """Start recording with database integration"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.camera_id}_{timestamp}.mp4"
            recording_path = os.path.join(RECORDINGS_DIR, filename)
            
            # Get video properties
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            recording_writer = cv2.VideoWriter(recording_path, fourcc, fps, (width, height))
            
            if recording_writer.isOpened():
                # Create recording entry in database
                with get_db_session() as db:
                    recording_create = RecordingInfoCreate(
                        camera_id=self.camera_id,
                        filename=filename,
                        file_path=recording_path,
                        resolution=f"{width}x{height}",
                        fps=fps,
                        output_directory=RECORDINGS_DIR
                    )
                    db_recording = RecordingService.create_recording(db, recording_create)
                    recording_id = db_recording.id
                
                logger.info(f"Started recording for camera {self.camera_id}: {recording_path}")
                return recording_writer, recording_path, recording_id
            else:
                logger.error(f"Failed to start recording for camera {self.camera_id}")
                return None, None, None
                
        except Exception as e:
            logger.error(f"Error starting recording for camera {self.camera_id}: {e}")
            return None, None, None
    
    def _stop_recording(self, recording_writer, recording_id):
        """Stop recording with database update"""
        try:
            recording_writer.release()
            
            if recording_id:
                # Update recording in database
                with get_db_session() as db:
                    # Get file size
                    db_recording = RecordingService.get_recording(db, recording_id)
                    if db_recording and os.path.exists(db_recording.file_path):
                        file_size = os.path.getsize(db_recording.file_path)
                        duration = (datetime.now() - db_recording.started_at).total_seconds()
                        
                        recording_update = RecordingInfoUpdate(
                            status="completed",
                            stopped_at=datetime.now(),
                            duration_seconds=duration,
                            file_size_bytes=file_size
                        )
                        RecordingService.update_recording(db, recording_id, recording_update)
                        
            logger.info(f"Stopped recording for camera {self.camera_id}")
            
        except Exception as e:
            logger.error(f"Error stopping recording for camera {self.camera_id}: {e}")

class CameraManager:
    """Enhanced camera manager with database integration"""
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
        self.camera_detections = {}  # For backward compatibility
        
        # Multi-model manager with database integration
        self.multi_model_manager = MultiModelManager()
        
        # Load available models from database
        self.available_models = self._load_available_models()
        
        # Load camera configurations from database
        self._load_camera_config()
    
    def set_event_loop(self, loop):
        """Set the event loop for async operations"""
        self.loop = loop
    
    def _load_available_models(self) -> Dict[str, Any]:
        """Load available models from database"""
        models = {}
        
        try:
            with get_db_session() as db:
                db_models = ModelService.get_models(db, enabled_only=True)
                for model in db_models:
                    models[model.id] = {
                        "id": model.id,
                        "name": model.name,
                        "path": model.path,
                        "version": model.version,
                        "description": model.description
                    }
        except Exception as e:
            logger.error(f"Failed to load models from database: {e}")
        
        # Add default models if none exist in database
        if not models:
            default_models = [
                {"id": "yolov8n", "name": "YOLOv8 Nano", "path": "./models/yolov8n.pt"},
                {"id": "yolov8s", "name": "YOLOv8 Small", "path": "./models/yolov8s.pt"},
                {"id": "yolov8m", "name": "YOLOv8 Medium", "path": "./models/yolov8m.pt"},
                {"id": "yolov8l", "name": "YOLOv8 Large", "path": "./models/yolov8l.pt"},
                {"id": "yolov8x", "name": "YOLOv8 Extra Large", "path": "./models/yolov8x.pt"},
            ]
            
            try:
                with get_db_session() as db:
                    for model_data in default_models:
                        model_create = ModelInfoCreate(**model_data)
                        ModelService.create_model(db, model_create)
                        models[model_data["id"]] = model_data
            except Exception as e:
                logger.error(f"Failed to create default models: {e}")
                # Fallback to in-memory models
                for model_data in default_models:
                    models[model_data["id"]] = model_data
        
        return models
    
    def _load_camera_config(self):
        """Load camera configurations from database"""
        try:
            with get_db_session() as db:
                db_cameras = CameraService.get_cameras(db)
                
                for db_camera in db_cameras:
                    camera = Camera.model_validate(db_camera)
                    self.cameras[camera.id] = camera
                    self.pre_recording_buffers[camera.id] = []
                    self.camera_status[camera.id] = db_camera.status.value
                    self.connected_clients[str(camera.id)] = []
        except Exception as e:
            logger.error(f"Failed to load cameras from database: {e}")
        
        # Load from JSON file if no cameras in database (backward compatibility)
        if not self.cameras:
            try:
                with open(CAMERA_CONFIG_PATH, 'r') as f:
                    config_data = json.load(f)
                    
                if isinstance(config_data, dict) and "cameras" in config_data:
                    camera_configs = config_data["cameras"]
                else:
                    camera_configs = config_data
                    
                with get_db_session() as db:
                    for camera_config in camera_configs:
                        camera_create = CameraCreate(**camera_config)
                        db_camera = CameraService.create_camera(db, camera_create)
                        
                        camera = Camera.model_validate(db_camera)
                        self.cameras[camera.id] = camera
                        self.pre_recording_buffers[camera.id] = []
                        self.camera_status[camera.id] = "disconnected"
                        self.connected_clients[str(camera.id)] = []
                        
            except FileNotFoundError:
                logger.info(f"Camera configuration file not found at {CAMERA_CONFIG_PATH}")
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON in camera configuration file")
            except Exception as e:
                logger.error(f"Failed to import cameras from JSON: {e}")
    
    def start_camera_threads(self):
        """Start camera processing threads for enabled cameras"""
        for camera_id, camera in self.cameras.items():
            if camera.enabled and camera_id not in self.camera_threads:
                camera_thread = CameraThread(camera_id, camera, self)
                self.camera_threads[camera_id] = camera_thread
                camera_thread.start()
    
    async def start_tracking_session(self, camera_id: str, request: TrackingRequest):
        """Start a new tracking session with database integration"""
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
            return TrackingResponse(
                session_id=session_id,
                camera_id=camera_id,
                model=model_id,
                message=f"Session already active"
            )
        
        # Create new detection session
        model_info = self.available_models[model_id]
        session = self.multi_model_manager.create_session(
            camera_id, model_id, model_info["path"], session_id
        )
        
        # Start processing for this session
        session.start_processing(self)
        
        # Enable detection for camera
        try:
            with get_db_session() as db:
                CameraService.update_camera(db, camera_id, CameraUpdate(detection_enabled=True))
            self.cameras[camera_id].detection_enabled = True
        except Exception as e:
            logger.error(f"Failed to update camera detection status: {e}")
        
        logger.info(f"Started tracking session {session_id} for camera {camera_id} with model {model_id}")
        
        return TrackingResponse(
            session_id=session_id,
            camera_id=camera_id,
            model=model_id,
            message=f"Tracking started for camera {camera_id} with model {model_id}"
        )
    
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
            try:
                with get_db_session() as db:
                    CameraService.update_camera(db, camera_id, CameraUpdate(detection_enabled=False))
                self.cameras[camera_id].detection_enabled = False
            except Exception as e:
                logger.error(f"Failed to update camera detection status: {e}")
        
        return {"message": message}
    
    def get_camera_detection_status(self, camera_id: str, model_id: str = None):
        """Get detection status for camera with database integration"""
        if camera_id not in self.cameras:
            raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")
        
        camera = self.cameras[camera_id]
        sessions = self.multi_model_manager.get_camera_sessions(camera_id)
        
        if model_id:
            sessions = [s for s in sessions if s.model_id == model_id]
        
        # Get recent detections from database
        last_detection = None
        recent_detections_count = 0
        total_detections_count = 0
        
        try:
            with get_db_session() as db:
                recent_time = datetime.now() - timedelta(minutes=1)
                detection_filter = DetectionFilter(
                    camera_id=camera_id,
                    start_date=recent_time
                )
                recent_detections = DetectionService.get_detections(db, detection_filter)
                recent_detections_count = len(recent_detections) if recent_detections else 0
                
                # Get total detections
                total_filter = DetectionFilter(camera_id=camera_id)
                total_detections = DetectionService.get_detections(db, total_filter)
                total_detections_count = len(total_detections) if total_detections else 0
                
                # Get last detection
                if recent_detections:
                    last_detection = Detection.model_validate(recent_detections[0])
        except Exception as e:
            logger.error(f"Failed to get detection data from database: {e}")
        
        # Aggregate detections from all sessions
        active_models = [s.model_id for s in sessions]
        
        return DetectionStatusResponse(
            camera_id=camera_id,
            detection_enabled=camera.detection_enabled,
            camera_status=self.camera_status.get(camera_id, "disconnected"),
            active_sessions=len(sessions),
            active_models=active_models,
            total_detections=total_detections_count,
            recent_detections=recent_detections_count,
            last_detection=last_detection,
            session_details=[
                {
                    "session_id": s.session_id,
                    "model": s.model_id,
                    "detections": len(s.detections),
                    "websocket_clients": len(s.websocket_clients)
                }
                for s in sessions
            ]
        )
    
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
    
    def get_cameras(self):
        """Get all configured cameras with their status from database"""
        try:
            with get_db_session() as db:
                db_cameras = CameraService.get_cameras(db)
                camera_list = []
                for db_camera in db_cameras:
                    camera_dict = Camera.model_validate(db_camera).model_dump()
                    camera_dict["status"] = self.camera_status.get(db_camera.id, db_camera.status.value)
                    camera_list.append(camera_dict)
                return camera_list
        except Exception as e:
            logger.error(f"Failed to get cameras from database: {e}")
            # Fallback to in-memory cameras
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

# Background task for system statistics collection
async def collect_system_stats():
    """Background task to collect and store system statistics"""
    while True:
        try:
            if camera_manager:
                # Collect system metrics
                memory_info = psutil.virtual_memory()
                cpu_percent = psutil.cpu_percent(interval=1)
                
                # Count active cameras and sessions
                active_cameras = len([s for s in camera_manager.camera_status.values() if s == "connected"])
                active_sessions = len(camera_manager.multi_model_manager.detection_sessions)
                
                # Get today's detection and alert counts from database
                with get_db_session() as db:
                    today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                    
                    # Count today's detections
                    detection_filter = DetectionFilter(start_date=today_start)
                    today_detections = DetectionService.get_detections(db, detection_filter)
                    
                    # Count today's alerts
                    alert_filter = AlertFilter(start_date=today_start)
                    today_alerts = AlertService.get_alerts(db, alert_filter)
                    
                    # Create system stats entry
                    stats_create = SystemStatsCreate(
                        active_cameras=active_cameras,
                        total_cameras=len(camera_manager.cameras),
                        active_sessions=active_sessions,
                        total_detections_today=len(today_detections) if today_detections else 0,
                        total_alerts_today=len(today_alerts) if today_alerts else 0,
                        active_recordings=len([c for c in camera_manager.cameras.values() if c.recording]),
                        memory_usage_mb=memory_info.used / 1024 / 1024,
                        cpu_usage_percent=cpu_percent,
                        websocket_connections=sum(len(clients) for clients in camera_manager.connected_clients.values()),
                        version="2.0.0"
                    )
                    
                    SystemStatsService.create_stats(db, stats_create)
            
        except Exception as e:
            logger.error(f"Error collecting system stats: {e}")
        
        # Sleep for 5 minutes before next collection
        await asyncio.sleep(300)

# Background task for database maintenance
async def database_maintenance():
    """Background task for database maintenance"""
    while True:
        try:
            if camera_manager:
                with get_db_session() as db:
                    # Run maintenance every 6 hours
                    maintenance_config = {
                        "detection_retention_days": 30,
                        "alert_retention_days": 90,
                        "session_cleanup_hours": 24,
                        "stats_retention_days": 90,
                        "activity_retention_days": 30
                    }
                    
                    results = DatabaseMaintenanceService.run_maintenance(db, maintenance_config)
                    logger.info(f"Database maintenance completed: {results}")
            
        except Exception as e:
            logger.error(f"Error during database maintenance: {e}")
        
        # Sleep for 6 hours before next maintenance
        await asyncio.sleep(21600)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    global camera_manager
    
    # Test database connection
    if not test_connection():
        logger.error("Database connection failed!")
        raise Exception("Database connection failed")
    
    # Initialize database
    init_database()
    logger.info("Database initialized successfully")
    
    # Initialize camera manager
    camera_manager = CameraManager()
    
    # Set event loop for camera manager
    loop = asyncio.get_event_loop()
    camera_manager.set_event_loop(loop)
    
    # Start camera threads
    camera_manager.start_camera_threads()
    
    # Start background tasks
    asyncio.create_task(collect_system_stats())
    asyncio.create_task(database_maintenance())
    
    logger.info("Application startup completed")
    
    yield
    
    # Shutdown logic
    if camera_manager:
        camera_manager.cleanup()
    
    logger.info("Application shutdown completed")

app = FastAPI(title="Multi-Camera YOLO Detection API with Database", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Activity logging middleware
@app.middleware("http")
async def log_activity_middleware(request, call_next):
    start_time = time.time()
    
    try:
        # Process request
        response = await call_next(request)
        
        # Calculate response time
        process_time = time.time() - start_time
        
        # Log activity to database (non-blocking)
        try:
            if camera_manager:
                # Extract relevant information
                action = f"{request.method}_{request.url.path.replace('/', '_').strip('_')}"
                resource_type = "api"
                
                # Extract resource ID from path if available
                resource_id = None
                path_parts = request.url.path.split('/')
                if len(path_parts) > 3 and path_parts[2] == "cameras":
                    resource_id = path_parts[3]
                
                activity_create = UserActivityCreate(
                    action=action,
                    resource_type=resource_type,
                    resource_id=resource_id,
                    endpoint=str(request.url.path),
                    method=request.method,
                    status_code=response.status_code,
                    user_agent=request.headers.get("user-agent"),
                    ip_address=request.client.host if request.client else None,
                    response_time_ms=process_time * 1000
                )
                
                # Log activity in background to avoid blocking
                with get_db_session() as db:
                    try:
                        UserActivityService.log_activity(db, activity_create)
                    except Exception as log_error:
                        logger.warning(f"Failed to log activity: {log_error}")
        
        except Exception as e:
            logger.error(f"Error in activity logging: {e}")
        
        return response
        
    except Exception as e:
        # If there's an error processing the request, still try to log it
        process_time = time.time() - start_time
        logger.error(f"Request processing error: {e}")
        
        # Return a generic error response
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"}
        )

# Enhanced API routes with database integration

@app.get("/api/cameras", response_model=List[Dict])
async def get_cameras():
    """Get all configured cameras"""
    return camera_manager.get_cameras()

@app.post("/api/cameras", response_model=Camera)
async def create_camera(camera: CameraCreate, db: Session = Depends(get_db)):
    """Create a new camera"""
    db_camera = CameraService.create_camera(db, camera)
    
    # Add to camera manager
    camera_obj = Camera.model_validate(db_camera)
    camera_manager.cameras[camera_obj.id] = camera_obj
    camera_manager.pre_recording_buffers[camera_obj.id] = []
    camera_manager.camera_status[camera_obj.id] = "disconnected"
    camera_manager.connected_clients[str(camera_obj.id)] = []
    
    # Start camera thread if enabled
    if camera_obj.enabled:
        camera_thread = CameraThread(camera_obj.id, camera_obj, camera_manager)
        camera_manager.camera_threads[camera_obj.id] = camera_thread
        camera_thread.start()
    
    return camera_obj

@app.put("/api/cameras/{camera_id}", response_model=Camera)
async def update_camera(camera_id: str, camera_update: CameraUpdate, db: Session = Depends(get_db)):
    """Update a camera"""
    db_camera = CameraService.update_camera(db, camera_id, camera_update)
    if not db_camera:
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")
    
    # Update in camera manager
    camera_obj = Camera.model_validate(db_camera)
    camera_manager.cameras[camera_id] = camera_obj
    
    return camera_obj

@app.delete("/api/cameras/{camera_id}")
async def delete_camera(camera_id: str, db: Session = Depends(get_db)):
    """Delete a camera"""
    # Stop camera thread if running
    if camera_id in camera_manager.camera_threads:
        camera_manager.camera_threads[camera_id].stop()
        del camera_manager.camera_threads[camera_id]
    
    # Stop all sessions for this camera
    sessions = camera_manager.multi_model_manager.get_camera_sessions(camera_id)
    for session in sessions:
        camera_manager.multi_model_manager.remove_session(session.session_id)
    
    # Delete from database
    success = CameraService.delete_camera(db, camera_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")
    
    # Remove from camera manager
    if camera_id in camera_manager.cameras:
        del camera_manager.cameras[camera_id]
    if camera_id in camera_manager.pre_recording_buffers:
        del camera_manager.pre_recording_buffers[camera_id]
    if camera_id in camera_manager.camera_status:
        del camera_manager.camera_status[camera_id]
    if str(camera_id) in camera_manager.connected_clients:
        del camera_manager.connected_clients[str(camera_id)]
    
    return {"message": f"Camera {camera_id} deleted successfully"}

@app.get("/api/models", response_model=List[ModelInfo])
async def get_models(db: Session = Depends(get_db)):
    """Get all available models"""
    db_models = ModelService.get_models(db)
    return [ModelInfo.model_validate(model) for model in db_models]

@app.post("/api/models", response_model=ModelInfo)
async def create_model(model: ModelInfoCreate, db: Session = Depends(get_db)):
    """Create a new model"""
    db_model = ModelService.create_model(db, model)
    
    # Update available models in camera manager
    camera_manager.available_models[model.id] = {
        "id": model.id,
        "name": model.name,
        "path": model.path,
        "version": model.version,
        "description": model.description
    }
    
    return ModelInfo.model_validate(db_model)

@app.put("/api/models/{model_id}", response_model=ModelInfo)
async def update_model(model_id: str, model_update: ModelInfoUpdate, db: Session = Depends(get_db)):
    """Update a model"""
    db_model = ModelService.update_model(db, model_id, model_update)
    if not db_model:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    
    # Update available models in camera manager
    camera_manager.available_models[model_id] = {
        "id": db_model.id,
        "name": db_model.name,
        "path": db_model.path,
        "version": db_model.version,
        "description": db_model.description
    }
    
    return ModelInfo.model_validate(db_model)

@app.post("/api/cameras/{camera_id}/start_tracking", response_model=TrackingResponse)
async def start_tracking(camera_id: Union[int, str], request: TrackingRequest):
    """Start tracking for a camera with specific model"""
    return await camera_manager.start_tracking_session(str(camera_id), request)

@app.post("/api/cameras/{camera_id}/start_tracking_multimodel", response_model=TrackingResponse)
async def start_tracking_multimodel(camera_id: Union[int, str], request: TrackingRequest):
    """Enhanced endpoint for multi-model tracking"""
    return await camera_manager.start_tracking_session(str(camera_id), request)

@app.post("/api/cameras/{camera_id}/stop_tracking")
async def stop_tracking(camera_id: Union[int, str], request: dict = None):
    """Stop tracking for a camera"""
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
async def stream_camera(camera_id: Union[int, str]):
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

@app.get("/api/cameras/{camera_id}/detection_status", response_model=DetectionStatusResponse)
async def get_detection_status(camera_id: Union[int, str], model: str = None):
    """Get real-time detection status for a camera"""
    return camera_manager.get_camera_detection_status(str(camera_id), model)

@app.get("/api/cameras/{camera_id}/multimodel_status", response_model=MultiModelStatusResponse)
async def get_multimodel_status(camera_id: Union[int, str]):
    """Get multi-model status for a camera"""
    sessions = camera_manager.multi_model_manager.get_camera_sessions(str(camera_id))
    
    return MultiModelStatusResponse(
        camera_id=camera_id,
        total_sessions=len(sessions),
        models=[s.model_id for s in sessions],
        active_models=list(set(s.model_id for s in sessions if s.active)),
        sessions=[
            {
                "session_id": s.session_id,
                "model": s.model_id,
                "active": s.active,
                "detections": len(s.detections),
                "websocket_clients": len(s.websocket_clients)
            }
            for s in sessions
        ]
    )

@app.get("/api/detections")
async def get_detections(
    camera_id: Optional[str] = None,
    model_id: Optional[str] = None,
    session_id: Optional[str] = None,
    class_name: Optional[str] = None,
    min_confidence: Optional[float] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    page: int = 1,
    page_size: int = 50,
    db: Session = Depends(get_db)
):
    """Get detections with filtering and pagination"""
    filters = DetectionFilter(
        camera_id=camera_id,
        model_id=model_id,
        session_id=session_id,
        class_name=class_name,
        min_confidence=min_confidence,
        start_date=start_date,
        end_date=end_date
    )
    pagination = PaginationParams(page=page, page_size=page_size)
    
    result = DetectionService.get_detections(db, filters, pagination)
    
    if isinstance(result, PaginatedResponse):
        # Convert to dict manually to avoid property serialization issues
        detections_data = []
        for item in result.items:
            detection_dict = {
                "id": item.id,
                "session_id": item.session_id,
                "camera_id": item.camera_id,
                "model_id": item.model_id,
                "bbox": [item.bbox_x1, item.bbox_y1, item.bbox_x2, item.bbox_y2],
                "bbox_x1": item.bbox_x1,
                "bbox_y1": item.bbox_y1,
                "bbox_x2": item.bbox_x2,
                "bbox_y2": item.bbox_y2,
                "confidence": item.confidence,
                "class_id": item.class_id,
                "class_name": item.class_name,
                "frame_width": item.frame_width,
                "frame_height": item.frame_height,
                "frame_timestamp": item.frame_timestamp.isoformat(),
                "inference_time": item.inference_time,
                "detected_at": item.detected_at.isoformat(),
                "meta_data": item.meta_data
            }
            detections_data.append(detection_dict)
        
        return {
            "items": detections_data,
            "total": result.total,
            "page": result.page,
            "page_size": result.page_size,
            "total_pages": result.total_pages,
            "has_next": result.has_next,
            "has_prev": result.has_prev
        }
    else:
        # Convert list to dict manually
        detections_data = []
        for item in result:
            detection_dict = {
                "id": item.id,
                "session_id": item.session_id,
                "camera_id": item.camera_id,
                "model_id": item.model_id,
                "bbox": [item.bbox_x1, item.bbox_y1, item.bbox_x2, item.bbox_y2],
                "bbox_x1": item.bbox_x1,
                "bbox_y1": item.bbox_y1,
                "bbox_x2": item.bbox_x2,
                "bbox_y2": item.bbox_y2,
                "confidence": item.confidence,
                "class_id": item.class_id,
                "class_name": item.class_name,
                "frame_width": item.frame_width,
                "frame_height": item.frame_height,
                "frame_timestamp": item.frame_timestamp.isoformat(),
                "inference_time": item.inference_time,
                "detected_at": item.detected_at.isoformat(),
                "meta_data": item.meta_data
            }
            detections_data.append(detection_dict)
        
        return detections_data

@app.get("/api/detections/statistics")
async def get_detection_statistics(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    camera_id: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get detection statistics"""
    return DetectionService.get_detection_statistics(db, start_date, end_date, camera_id)

@app.get("/api/alerts")
async def get_alerts(
    camera_id: Optional[str] = None,
    model_id: Optional[str] = None,
    session_id: Optional[str] = None,
    object_type: Optional[str] = None,
    acknowledged: Optional[bool] = None,
    min_confidence: Optional[float] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    page: int = 1,
    page_size: int = 50,
    db: Session = Depends(get_db)
):
    """Get alerts with filtering and pagination"""
    filters = AlertFilter(
        camera_id=camera_id,
        model_id=model_id,
        session_id=session_id,
        object_type=object_type,
        acknowledged=acknowledged,
        min_confidence=min_confidence,
        start_date=start_date,
        end_date=end_date
    )
    pagination = PaginationParams(page=page, page_size=page_size)
    
    result = AlertService.get_alerts(db, filters, pagination)
    
    if isinstance(result, PaginatedResponse):
        # Convert to dict manually to avoid property serialization issues
        alerts_data = []
        for item in result.items:
            alert_dict = {
                "id": item.id,
                "detection_id": item.detection_id,
                "session_id": item.session_id,
                "camera_id": item.camera_id,
                "model_id": item.model_id,
                "object_type": item.object_type,
                "confidence": item.confidence,
                "bbox": [item.bbox_x1, item.bbox_y1, item.bbox_x2, item.bbox_y2],
                "bbox_x1": item.bbox_x1,
                "bbox_y1": item.bbox_y1,
                "bbox_x2": item.bbox_x2,
                "bbox_y2": item.bbox_y2,
                "acknowledged": item.acknowledged,
                "acknowledged_at": item.acknowledged_at.isoformat() if item.acknowledged_at else None,
                "acknowledged_by": item.acknowledged_by,
                "image_data": item.image_data,
                "image_size": item.image_size,
                "triggered_at": item.triggered_at.isoformat(),
                "notification_sent": item.notification_sent,
                "notification_sent_at": item.notification_sent_at.isoformat() if item.notification_sent_at else None,
                "notification_channels": item.notification_channels
            }
            alerts_data.append(alert_dict)
        
        return {
            "items": alerts_data,
            "total": result.total,
            "page": result.page,
            "page_size": result.page_size,
            "total_pages": result.total_pages,
            "has_next": result.has_next,
            "has_prev": result.has_prev
        }
    else:
        # Convert list to dict manually
        alerts_data = []
        for item in result:
            alert_dict = {
                "id": item.id,
                "detection_id": item.detection_id,
                "session_id": item.session_id,
                "camera_id": item.camera_id,
                "model_id": item.model_id,
                "object_type": item.object_type,
                "confidence": item.confidence,
                "bbox": [item.bbox_x1, item.bbox_y1, item.bbox_x2, item.bbox_y2],
                "bbox_x1": item.bbox_x1,
                "bbox_y1": item.bbox_y1,
                "bbox_x2": item.bbox_x2,
                "bbox_y2": item.bbox_y2,
                "acknowledged": item.acknowledged,
                "acknowledged_at": item.acknowledged_at.isoformat() if item.acknowledged_at else None,
                "acknowledged_by": item.acknowledged_by,
                "image_data": item.image_data,
                "image_size": item.image_size,
                "triggered_at": item.triggered_at.isoformat(),
                "notification_sent": item.notification_sent,
                "notification_sent_at": item.notification_sent_at.isoformat() if item.notification_sent_at else None,
                "notification_channels": item.notification_channels
            }
            alerts_data.append(alert_dict)
        
        return alerts_data

@app.post("/api/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str, acknowledged_by: str = None, db: Session = Depends(get_db)):
    """Acknowledge an alert"""
    success = AlertService.acknowledge_alert(db, alert_id, acknowledged_by)
    if not success:
        raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")
    
    return {"message": f"Alert {alert_id} acknowledged successfully"}

@app.get("/api/alerts/statistics")
async def get_alert_statistics(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    camera_id: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get alert statistics"""
    return AlertService.get_alert_statistics(db, start_date, end_date, camera_id)

@app.get("/api/recordings")
async def get_recordings(
    camera_id: Optional[str] = None,
    status: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    min_duration: Optional[float] = None,
    max_duration: Optional[float] = None,
    page: int = 1,
    page_size: int = 50,
    db: Session = Depends(get_db)
):
    """Get recordings with filtering and pagination"""
    filters = RecordingFilter(
        camera_id=camera_id,
        status=status,
        start_date=start_date,
        end_date=end_date,
        min_duration=min_duration,
        max_duration=max_duration
    )
    pagination = PaginationParams(page=page, page_size=page_size)
    
    result = RecordingService.get_recordings(db, filters, pagination)
    
    if isinstance(result, PaginatedResponse):
        result.items = [RecordingInfo.model_validate(item) for item in result.items]
        return result
    else:
        return [RecordingInfo.model_validate(item) for item in result]

@app.post("/api/cameras/{camera_id}/start_recording")
async def start_recording(camera_id: Union[int, str], request: dict = None, db: Session = Depends(get_db)):
    """Start recording for a specific camera"""
    if str(camera_id) not in camera_manager.cameras:
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")
    
    # Check if already recording
    camera = camera_manager.cameras[str(camera_id)]
    if camera.recording:
        raise HTTPException(status_code=400, detail=f"Camera {camera_id} is already recording")
    
    # Update camera recording status
    CameraService.update_camera(db, str(camera_id), CameraUpdate(recording=True))
    camera_manager.cameras[str(camera_id)].recording = True
    
    return {"message": f"Recording started for camera {camera_id}"}

@app.post("/api/cameras/{camera_id}/stop_recording")
async def stop_recording(camera_id: Union[int, str], db: Session = Depends(get_db)):
    """Stop recording for a specific camera"""
    if str(camera_id) not in camera_manager.cameras:
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")
    
    # Update camera recording status
    CameraService.update_camera(db, str(camera_id), CameraUpdate(recording=False))
    camera_manager.cameras[str(camera_id)].recording = False
    
    return {"message": f"Recording stopped for camera {camera_id}"}

@app.delete("/api/recordings/{recording_id}")
async def delete_recording(recording_id: str, db: Session = Depends(get_db)):
    """Delete a specific recording"""
    success = RecordingService.delete_recording(db, recording_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Recording {recording_id} not found")
    
    return {"message": f"Recording {recording_id} deleted successfully"}

@app.get("/api/recordings/{recording_id}/download")
async def download_recording(recording_id: str, db: Session = Depends(get_db)):
    """Download a specific recording file"""
    recording = RecordingService.get_recording(db, recording_id)
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

@app.get("/api/sessions")
async def get_sessions(camera_id: str = None, model_id: str = None, db: Session = Depends(get_db)):
    """Get detection sessions"""
    if camera_id:
        db_sessions = SessionService.get_camera_sessions(db, camera_id)
    elif model_id:
        db_sessions = SessionService.get_model_sessions(db, model_id)
    else:
        # Get all active sessions
        db_sessions = db.query(DetectionSessionModel).filter(DetectionSessionModel.active == True).all()
    
    return [DetectionSession.model_validate(session) for session in db_sessions]

@app.get("/api/system/stats", response_model=SystemStats)
async def get_system_stats(db: Session = Depends(get_db)):
    """Get latest system statistics"""
    latest_stats = SystemStatsService.get_latest_stats(db)
    if not latest_stats:
        raise HTTPException(status_code=404, detail="No system statistics available")
    
    return SystemStats.model_validate(latest_stats)

@app.get("/api/system/stats/history")
async def get_system_stats_history(
    hours: int = 24,
    page: int = 1,
    page_size: int = 50,
    db: Session = Depends(get_db)
):
    """Get system statistics history"""
    pagination = PaginationParams(page=page, page_size=page_size)
    result = SystemStatsService.get_stats_history(db, hours, pagination)
    
    if isinstance(result, PaginatedResponse):
        result.items = [SystemStats.model_validate(item) for item in result.items]
        return result
    else:
        return [SystemStats.model_validate(item) for item in result]

@app.get("/api/system/activities")
async def get_user_activities(
    action: str = None,
    resource_type: str = None,
    hours: int = 24,
    page: int = 1,
    page_size: int = 50,
    db: Session = Depends(get_db)
):
    """Get user activities"""
    pagination = PaginationParams(page=page, page_size=page_size)
    result = UserActivityService.get_activities(db, action, resource_type, hours, pagination)
    
    if isinstance(result, PaginatedResponse):
        result.items = [UserActivity.model_validate(item) for item in result.items]
        return result
    else:
        return [UserActivity.model_validate(item) for item in result]

@app.get("/api/system/maintenance")
async def run_database_maintenance(db: Session = Depends(get_db)):
    """Run database maintenance operations"""
    results = DatabaseMaintenanceService.run_maintenance(db)
    return {"message": "Database maintenance completed", "results": results}

@app.get("/api/system/database/info")
async def get_database_info(db: Session = Depends(get_db)):
    """Get database size and information"""
    return DatabaseMaintenanceService.get_database_size_info(db)

@app.post("/api/system/database/optimize")
async def optimize_database(db: Session = Depends(get_db)):
    """Optimize database tables"""
    return DatabaseMaintenanceService.optimize_tables(db)

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
    """WebSocket endpoint for camera alerts"""
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
                        "data": status.model_dump()
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
async def websocket_model_specific_alerts(websocket: WebSocket, camera_id: Union[int, str], model: str):
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
                        "data": status.model_dump()
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

@app.get("/api/health", response_model=HealthResponse)
async def health_check(db: Session = Depends(get_db)):
    """Enhanced health check with database connectivity"""
    try:
        # Test database connectivity
        db.execute("SELECT 1")
        database_connected = True
    except Exception:
        database_connected = False
    
    # Get recording count from database
    total_recordings = 0
    try:
        recordings = RecordingService.get_recordings(db)
        total_recordings = len(recordings) if recordings else 0
    except Exception:
        pass
    
    return HealthResponse(
        status="ok" if database_connected else "degraded",
        active_cameras=len([s for s in camera_manager.camera_status.values() if s == "connected"]),
        total_cameras=len(camera_manager.cameras),
        multi_model_support=True,
        active_sessions=len(camera_manager.multi_model_manager.detection_sessions),
        loaded_models=len(camera_manager.multi_model_manager.loaded_models),
        recording_support=True,
        cameras_recording=len([c for c in camera_manager.cameras.values() if c.recording]),
        total_recordings=total_recordings,
        database_connected=database_connected
    )

@app.get("/api/detection/health", response_model=DetectionHealthResponse)
async def detection_health(db: Session = Depends(get_db)):
    """Get overall detection system health with database info"""
    try:
        db.execute("SELECT 1")
        database_connected = True
    except Exception:
        database_connected = False
    
    active_cameras = sum(1 for status in camera_manager.camera_status.values() 
                        if status == "connected")
    
    detection_enabled_cameras = sum(1 for cam in camera_manager.cameras.values() 
                                   if cam.detection_enabled)
    
    total_connections = sum(len(clients) for clients in camera_manager.connected_clients.values())
    
    # Add session-specific connections
    total_session_connections = sum(len(session.websocket_clients) 
                                   for session in camera_manager.multi_model_manager.detection_sessions.values())
    
    return DetectionHealthResponse(
        multi_model_support=True,
        loaded_models=list(camera_manager.multi_model_manager.loaded_models.keys()),
        total_cameras=len(camera_manager.cameras),
        active_cameras=active_cameras,
        detection_enabled_cameras=detection_enabled_cameras,
        active_sessions=len(camera_manager.multi_model_manager.detection_sessions),
        total_websocket_connections=total_connections + total_session_connections,
        system_status="healthy" if database_connected else "degraded",
        database_connected=database_connected
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)