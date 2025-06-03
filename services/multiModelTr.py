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
from collections import defaultdict
import copy

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
MAX_WORKERS = 6

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

class DetectionModel(BaseModel):
    id: str
    name: str
    path: str
    
class Alert(BaseModel):
    camera_id: Union[int, str]
    timestamp: str
    object_type: str
    confidence: float
    bbox: List[float]
    image_data: Optional[str] = None

class ModelDetectionRequest(BaseModel):
    model_id: str
    camera_id: Union[int, str]
    settings: Optional[Dict] = {}

class MultiModelDetectionResult(BaseModel):
    camera_id: Union[int, str]
    model_id: str
    detections: List[Dict]
    timestamp: str
    frame_id: str

class CameraThread:
    """Optimized camera processing thread with frame sharing capability"""
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
        self.frame_subscribers = set()  # Track who needs frames
        
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
        
    def subscribe_to_frames(self, subscriber_id):
        """Subscribe to frame updates"""
        self.frame_subscribers.add(subscriber_id)
        logger.info(f"Subscriber {subscriber_id} added to camera {self.camera_id}")
        
    def unsubscribe_from_frames(self, subscriber_id):
        """Unsubscribe from frame updates"""
        self.frame_subscribers.discard(subscriber_id)
        logger.info(f"Subscriber {subscriber_id} removed from camera {self.camera_id}")
        
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
        """Main camera processing loop with frame distribution"""
        if not self._initialize_camera():
            return
            
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        frame_interval = 1.0 / fps
        pre_recording_frames_count = int(fps * PRE_RECORDING_BUFFER_SECONDS)
        
        recording_writer = None
        recording_path = None
        frame_counter = 0
        
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
                frame_counter += 1
                frame_id = f"{self.camera_id}_{frame_counter}_{int(current_time * 1000)}"
                
                # Store latest frame with metadata
                frame_data = {
                    'frame': frame.copy(),
                    'timestamp': current_time,
                    'frame_id': frame_id
                }
                self.camera_manager.camera_frames[self.camera_id] = frame_data
                
                # Update pre-recording buffer
                pre_buffer = self.camera_manager.pre_recording_buffers[self.camera_id]
                pre_buffer.append(frame.copy())
                if len(pre_buffer) > pre_recording_frames_count:
                    pre_buffer.pop(0)
                
                # Distribute frame to all detection processors
                self.camera_manager.multi_model_processor.distribute_frame(
                    self.camera_id, frame.copy(), frame_id, current_time
                )
                
                # Original single model detection (for backward compatibility)
                if (self.camera_manager.detection_model and 
                    self.camera.detection_enabled):
                    self.camera_manager.single_model_processor.queue_frame(
                        self.camera_id, frame.copy(), current_time
                    )
                
                # Handle recording
                if self.camera.recording:
                    if recording_writer is None:
                        recording_writer, recording_path = self._start_recording()
                    
                    if recording_writer:
                        # Write pre-recording buffer first
                        if recording_path and len(pre_buffer) > 0:
                            for buffer_frame in pre_buffer:
                                recording_writer.write(buffer_frame)
                            pre_buffer.clear()
                        
                        recording_writer.write(frame)
                        
                elif recording_writer is not None:
                    recording_writer.release()
                    recording_writer = None
                    recording_path = None
                
                # Broadcast frame to WebSocket clients (async)
                asyncio.run_coroutine_threadsafe(
                    self.camera_manager._broadcast_frame_async(self.camera_id, frame),
                    self.camera_manager.loop
                )
                
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

class SingleModelDetectionProcessor:
    """Original single model detection processor for backward compatibility"""
    def __init__(self, camera_manager):
        self.camera_manager = camera_manager
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.active = True
        self.processing_thread = None
        self.frame_queue = queue.Queue(maxsize=50)
        
    def start(self):
        """Start the detection processing thread"""
        self.processing_thread = threading.Thread(target=self._process_detections, daemon=True)
        self.processing_thread.start()
        logger.info("Single model detection processor started")
        
    def stop(self):
        """Stop the detection processor"""
        self.active = False
        self.executor.shutdown(wait=True)
        if self.processing_thread:
            self.processing_thread.join(timeout=2)
        logger.info("Single model detection processor stopped")
        
    def queue_frame(self, camera_id, frame, timestamp):
        """Queue a frame for detection"""
        try:
            if not self.frame_queue.full():
                self.frame_queue.put_nowait((camera_id, frame, timestamp))
        except queue.Full:
            pass  # Skip if queue is full
            
    def _process_detections(self):
        """Process detection queue"""
        while self.active:
            try:
                if not self.frame_queue.empty():
                    camera_id, frame, timestamp = self.frame_queue.get(timeout=0.1)
                    self._process_single_frame(camera_id, frame, timestamp)
                else:
                    time.sleep(0.01)
                    
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in single model detection processor: {e}")
                time.sleep(0.1)
    
    def _process_single_frame(self, camera_id, frame, timestamp):
        """Process a single frame for the main detection model"""
        try:
            if not self.camera_manager.detection_model:
                return
                
            camera = self.camera_manager.cameras.get(camera_id)
            if not camera or not camera.detection_enabled:
                return
                
            # Run detection
            results = self.camera_manager.detection_model.predict(
                frame, conf=0.25, verbose=False
            )
            
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
                            "timestamp": datetime.fromtimestamp(timestamp).isoformat()
                        }
                        detections.append(detection)
                        
                        # Send alert if confidence exceeds threshold
                        if confidence > camera.alert_threshold:
                            self._send_alert(camera_id, detection, frame)
            
            # Update detections for backward compatibility
            self.camera_manager.camera_detections[camera_id] = detections
                
        except Exception as e:
            logger.error(f"Error processing frame for camera {camera_id}: {e}")
    
    def _send_alert(self, camera_id, detection, frame):
        """Send an alert for high-confidence detections"""
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

            alert = Alert(
                camera_id=camera_id,
                timestamp=detection["timestamp"],
                object_type=detection["class_name"],
                confidence=detection["confidence"],
                bbox=detection["bbox"],
                image_data=img_base64
            )

            logger.info(f"🚨 ALERT: {alert.object_type} detected on camera {camera_id} with {alert.confidence:.1%} confidence")

            # Broadcast alert asynchronously
            asyncio.run_coroutine_threadsafe(
                self.camera_manager._broadcast_alert_async(camera_id, alert),
                self.camera_manager.loop
            )
            
        except Exception as e:
            logger.error(f"Error sending alert for camera {camera_id}: {e}")

class MultiModelDetectionProcessor:
    """Enhanced processor for handling multiple models simultaneously"""
    def __init__(self, camera_manager):
        self.camera_manager = camera_manager
        self.active = True
        self.loaded_models = {}  # model_id -> YOLO model instance
        self.model_threads = {}  # model_id -> processing thread
        self.frame_queues = {}   # model_id -> frame queue
        self.detection_results = defaultdict(lambda: defaultdict(list))  # camera_id -> model_id -> detections
        self.model_subscribers = defaultdict(set)  # model_id -> set of subscriber_ids
        self.executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
        
    def start(self):
        """Start the multi-model detection processor"""
        logger.info("Multi-model detection processor started")
        
    def stop(self):
        """Stop all model processing threads"""
        self.active = False
        self.executor.shutdown(wait=True)
        
        for thread in self.model_threads.values():
            if thread and thread.is_alive():
                thread.join(timeout=1)
                
        logger.info("Multi-model detection processor stopped")
        
    def load_model(self, model_id, model_path):
        """Load a YOLO model for multi-model detection"""
        try:
            if model_id in self.loaded_models:
                logger.info(f"Model {model_id} already loaded")
                return True
                
            # Load model in executor to avoid blocking
            def _load():
                model = YOLO(model_path)
                return model
                
            loop = asyncio.get_event_loop()
            model = loop.run_in_executor(self.executor, _load)
            
            # This is a simplified version - in production, you'd want to handle this properly
            self.loaded_models[model_id] = YOLO(model_path)
            self.frame_queues[model_id] = queue.Queue(maxsize=100)
            
            # Start processing thread for this model
            thread = threading.Thread(
                target=self._process_model_detections, 
                args=(model_id,), 
                daemon=True
            )
            thread.start()
            self.model_threads[model_id] = thread
            
            logger.info(f"Model {model_id} loaded and processing thread started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            return False
            
    def unload_model(self, model_id):
        """Unload a model and stop its processing"""
        if model_id in self.loaded_models:
            del self.loaded_models[model_id]
            
        if model_id in self.frame_queues:
            del self.frame_queues[model_id]
            
        if model_id in self.model_threads:
            # Thread will stop when self.active becomes False or model is removed
            del self.model_threads[model_id]
            
        if model_id in self.model_subscribers:
            del self.model_subscribers[model_id]
            
        logger.info(f"Model {model_id} unloaded")
        
    def subscribe_to_model(self, model_id, subscriber_id):
        """Subscribe to detections from a specific model"""
        self.model_subscribers[model_id].add(subscriber_id)
        logger.info(f"Subscriber {subscriber_id} added to model {model_id}")
        
    def unsubscribe_from_model(self, model_id, subscriber_id):
        """Unsubscribe from model detections"""
        self.model_subscribers[model_id].discard(subscriber_id)
        logger.info(f"Subscriber {subscriber_id} removed from model {model_id}")
        
    def distribute_frame(self, camera_id, frame, frame_id, timestamp):
        """Distribute a frame to all loaded models that have subscribers"""
        for model_id in self.loaded_models:
            if self.model_subscribers[model_id]:  # Only if someone is subscribed
                try:
                    if not self.frame_queues[model_id].full():
                        self.frame_queues[model_id].put_nowait({
                            'camera_id': camera_id,
                            'frame': frame.copy(),
                            'frame_id': frame_id,
                            'timestamp': timestamp
                        })
                except queue.Full:
                    pass  # Skip if queue is full
                    
    def _process_model_detections(self, model_id):
        """Process detections for a specific model"""
        logger.info(f"Started detection processing for model {model_id}")
        
        while self.active and model_id in self.loaded_models:
            try:
                if model_id not in self.frame_queues:
                    time.sleep(0.1)
                    continue
                    
                frame_queue = self.frame_queues[model_id]
                
                if not frame_queue.empty():
                    frame_data = frame_queue.get(timeout=0.1)
                    self._process_frame_with_model(model_id, frame_data)
                else:
                    time.sleep(0.01)
                    
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in model {model_id} processing: {e}")
                time.sleep(0.1)
                
        logger.info(f"Stopped detection processing for model {model_id}")
        
    def _process_frame_with_model(self, model_id, frame_data):
        """Process a single frame with a specific model"""
        try:
            model = self.loaded_models[model_id]
            camera_id = frame_data['camera_id']
            frame = frame_data['frame']
            frame_id = frame_data['frame_id']
            timestamp = frame_data['timestamp']
            
            # Run detection
            results = model.predict(frame, conf=0.25, verbose=False)
            
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
                            "frame_id": frame_id
                        }
                        detections.append(detection)
            
            # Store results
            self.detection_results[camera_id][model_id] = detections
            
            # Broadcast to subscribers if needed
            if self.model_subscribers[model_id]:
                result_data = MultiModelDetectionResult(
                    camera_id=camera_id,
                    model_id=model_id,
                    detections=detections,
                    timestamp=datetime.fromtimestamp(timestamp).isoformat(),
                    frame_id=frame_id
                )
                
                # You can add WebSocket broadcasting here if needed
                logger.debug(f"Processed frame {frame_id} for model {model_id}: {len(detections)} detections")
                
        except Exception as e:
            logger.error(f"Error processing frame with model {model_id}: {e}")
            
    def get_detections(self, camera_id, model_id):
        """Get latest detections for a camera-model combination"""
        return self.detection_results[camera_id].get(model_id, [])
        
    def get_all_model_detections(self, camera_id):
        """Get detections from all models for a camera"""
        return dict(self.detection_results[camera_id])

class CameraManager:
    def __init__(self):
        self.cameras: Dict[Union[int, str], Camera] = {}
        self.camera_threads: Dict[Union[int, str], CameraThread] = {}
        self.camera_frames: Dict[Union[int, str], Dict] = {}  # Now stores frame with metadata
        self.camera_detections: Dict[Union[int, str], List] = {}  # For backward compatibility
        self.camera_status: Dict[Union[int, str], str] = {}
        self.pre_recording_buffers: Dict[Union[int, str], List] = {}
        self.detection_model = None  # Main model for backward compatibility
        self.current_model_id = None
        self.active = True
        self.connected_clients: Dict[str, List[WebSocket]] = {}
        self.loop = None
        
        # Initialize processors
        self.single_model_processor = SingleModelDetectionProcessor(self)
        self.multi_model_processor = MultiModelDetectionProcessor(self)
        
        # Load available models
        self.available_models = self._load_available_models()
        
        # Load camera configurations
        self._load_camera_config()
        
        # Start processors
        self.single_model_processor.start()
        self.multi_model_processor.start()
    
    def set_event_loop(self, loop):
        """Set the event loop for async operations"""
        self.loop = loop
    
    def _load_available_models(self) -> Dict[str, DetectionModel]:
        models = {}
        default_models = [
            {"id": "yolov8n", "name": "YOLOv8 Nano", "path": "./models/yolov8n.pt"},
            {"id": "yolov8s", "name": "YOLOv8 Small", "path": "./models/yolov8s.pt"},
            {"id": "yolov8m", "name": "YOLOv8 Medium", "path": "./models/yolov8m.pt"},
            {"id": "yolov8l", "name": "YOLOv8 Large", "path": "./models/yolov8l.pt"},
        ]
        
        for model in default_models:
            models[model["id"]] = DetectionModel(**model)
            
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
                self.camera_detections[camera_id] = []
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
    
    # Multi-model specific methods
    async def load_detection_model_for_preview(self, model_id, subscriber_id=None):
        """Load a model for multi-model preview"""
        if model_id not in self.available_models:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
        
        model = self.available_models[model_id]
        success = self.multi_model_processor.load_model(model_id, model.path)
        
        if success and subscriber_id:
            self.multi_model_processor.subscribe_to_model(model_id, subscriber_id)
            
        return {"message": f"Model {model.name} loaded for preview", "success": success}
    
    def unload_detection_model_for_preview(self, model_id, subscriber_id=None):
        """Unload a model from multi-model preview"""
        if subscriber_id:
            self.multi_model_processor.unsubscribe_from_model(model_id, subscriber_id)
            
        # Only unload if no more subscribers
        if not self.multi_model_processor.model_subscribers[model_id]:
            self.multi_model_processor.unload_model(model_id)
            
        return {"message": f"Model {model_id} unloaded from preview"}
    
    def get_multi_model_detections(self, camera_id, model_id=None):
        """Get detections from multi-model processor"""
        if model_id:
            return self.multi_model_processor.get_detections(camera_id, model_id)
        else:
            return self.multi_model_processor.get_all_model_detections(camera_id)
    
    def get_model_detection_frame_generator(self, camera_id, model_id):
        """Get a generator for frames with specific model detections"""
        if camera_id not in self.cameras:
            raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")
        
        def generate():
            while self.active:
                if camera_id in self.camera_frames:
                    frame_data = self.camera_frames[camera_id]
                    frame = frame_data['frame'].copy()
                    
                    # Draw detections from specific model
                    detections = self.multi_model_processor.get_detections(camera_id, model_id)
                    for det in detections:
                        x1, y1, x2, y2 = map(int, det["bbox"])
                        
                        # Different colors for different models
                        color = self._get_model_color(model_id)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Add model info to label
                        label = f"{det['class_name']}: {det['confidence']:.2f} ({model_id})"
                        cv2.putText(frame, label, (x1, y1 - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # Encode frame as JPEG
                    _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
                else:
                    # If no frame is available, return a black frame
                    black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    _, jpeg = cv2.imencode('.jpg', black_frame)
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
                
                time.sleep(0.033)  # ~30 FPS
        
        return generate
    
    def _get_model_color(self, model_id):
        """Get a unique color for each model"""
        colors = {
            'yolov8n': (0, 255, 0),    # Green
            'yolov8s': (255, 0, 0),    # Blue
            'yolov8m': (0, 0, 255),    # Red
            'yolov8l': (255, 255, 0),  # Cyan
            'yolov8x': (255, 0, 255),  # Magenta
        }
        return colors.get(model_id, (128, 128, 128))  # Default gray
    
    # Original methods for backward compatibility
    async def _broadcast_alert_async(self, camera_id, alert):
        """Broadcast alert to WebSocket clients asynchronously"""
        str_camera_id = str(camera_id)
        
        if str_camera_id in self.connected_clients and self.connected_clients[str_camera_id]:
            alert_message = {
                "type": "alert",
                "camera_id": str(camera_id),
                "alert": {
                    "id": f"{camera_id}_{int(time.time() * 1000)}_{random.randint(1000, 9999)}",
                    "object_type": alert.object_type,
                    "confidence": alert.confidence,
                    "bbox": alert.bbox,
                    "timestamp": alert.timestamp,
                    "image_data": alert.image_data
                },
                "metadata": {
                    "detection_count": len(self.camera_detections.get(camera_id, [])),
                    "camera_status": self.camera_status.get(camera_id, "unknown"),
                    "model_id": self.current_model_id
                }
            }
            
            disconnected_clients = []
            
            for client in self.connected_clients[str_camera_id]:
                try:
                    await client.send_json(alert_message)
                except Exception as e:
                    logger.error(f"Failed to send alert to client: {e}")
                    disconnected_clients.append(client)
            
            # Clean up disconnected clients
            for client in disconnected_clients:
                if client in self.connected_clients[str_camera_id]:
                    self.connected_clients[str_camera_id].remove(client)

    async def _broadcast_frame_async(self, camera_id, frame):
        """Broadcast frame to WebSocket clients asynchronously"""
        str_camera_id = str(camera_id)
        if str_camera_id in self.connected_clients and self.connected_clients[str_camera_id]:
            try:
                # Encode frame as JPEG
                _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
                frame_data = jpeg.tobytes()
                
                disconnected_clients = []
                for client in self.connected_clients[str_camera_id]:
                    try:
                        await client.send_bytes(frame_data)
                    except Exception:
                        disconnected_clients.append(client)
                
                # Remove disconnected clients
                for client in disconnected_clients:
                    if client in self.connected_clients[str_camera_id]:
                        self.connected_clients[str_camera_id].remove(client)
            except Exception as e:
                logger.error(f"Error broadcasting frame for camera {camera_id}: {e}")
    
    async def register_client(self, camera_id, websocket):
        """Register a WebSocket client for a camera"""
        str_camera_id = str(camera_id)
        if str_camera_id not in self.connected_clients:
            self.connected_clients[str_camera_id] = []
        self.connected_clients[str_camera_id].append(websocket)
        logger.info(f"Registered WebSocket client for camera {camera_id}")
    
    async def unregister_client(self, camera_id, websocket):
        """Unregister a WebSocket client"""
        str_camera_id = str(camera_id)
        if str_camera_id in self.connected_clients and websocket in self.connected_clients[str_camera_id]:
            self.connected_clients[str_camera_id].remove(websocket)
            logger.info(f"Unregistered WebSocket client for camera {camera_id}")
    
    async def load_model(self, model_id):
        """Load a YOLO model for detection (backward compatibility)"""
        if model_id not in self.available_models:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
        
        model = self.available_models[model_id]
        
        def _load_model():
            try:
                self.detection_model = YOLO(model.path)
                self.current_model_id = model_id
                logger.info(f"Loaded main model: {model.name}")
            except Exception as e:
                logger.error(f"Failed to load model {model_id}: {e}")
                raise
            
        # Load model in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _load_model)
        
        return {"message": f"Model {model.name} loaded successfully"}
    
    # Standard CRUD operations (unchanged)
    def get_cameras(self):
        """Get all configured cameras with their status"""
        camera_list = []
        for camera_id, camera in self.cameras.items():
            camera_dict = camera.model_dump()
            camera_dict["status"] = self.camera_status.get(camera_id, "disconnected")
            camera_list.append(camera_dict)
        return camera_list
    
    def get_camera(self, camera_id):
        """Get a specific camera by ID with status"""
        if camera_id not in self.cameras:
            raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")
        
        camera_dict = self.cameras[camera_id].model_dump()
        camera_dict["status"] = self.camera_status.get(camera_id, "disconnected")
        return camera_dict
    
    def add_camera(self, camera_data):
        """Add a new camera"""
        camera_id = camera_data.get("id")
        if not camera_id:
            camera_id = str(uuid.uuid4())
            camera_data["id"] = camera_id
            
        camera = Camera(**camera_data)
        self.cameras[camera_id] = camera
        self.camera_detections[camera_id] = []
        self.pre_recording_buffers[camera_id] = []
        self.camera_status[camera_id] = "disconnected"
        self.connected_clients[str(camera_id)] = []
        
        # Start the camera thread if enabled
        if camera.enabled:
            camera_thread = CameraThread(camera_id, camera, self)
            self.camera_threads[camera_id] = camera_thread
            camera_thread.start()
        
        self._save_camera_config()
        
        camera_dict = camera.model_dump()
        camera_dict["status"] = self.camera_status.get(camera_id, "disconnected")
        return camera_dict
    
    def update_camera(self, camera_id, camera_data):
        """Update an existing camera"""
        if camera_id not in self.cameras:
            raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")
        
        current_camera = self.cameras[camera_id]
        was_enabled = current_camera.enabled
        
        camera_data["id"] = camera_id
        updated_camera = Camera(**camera_data)
        self.cameras[camera_id] = updated_camera
        
        # Handle changes to enabled status
        if not was_enabled and updated_camera.enabled:
            # Start camera thread
            if camera_id not in self.camera_threads:
                camera_thread = CameraThread(camera_id, updated_camera, self)
                self.camera_threads[camera_id] = camera_thread
                camera_thread.start()
        elif was_enabled and not updated_camera.enabled:
            # Stop camera thread
            if camera_id in self.camera_threads:
                self.camera_threads[camera_id].stop()
                del self.camera_threads[camera_id]
                self.camera_status[camera_id] = "disconnected"
        
        self._save_camera_config()
        
        camera_dict = updated_camera.model_dump()
        camera_dict["status"] = self.camera_status.get(camera_id, "disconnected")
        return camera_dict
    
    def delete_camera(self, camera_id):
        """Delete a camera"""
        if camera_id not in self.cameras:
            raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")
        
        # Stop camera thread if running
        if camera_id in self.camera_threads:
            self.camera_threads[camera_id].stop()
            del self.camera_threads[camera_id]
        
        # Remove camera from dictionaries
        del self.cameras[camera_id]
        self.camera_detections.pop(camera_id, None)
        self.pre_recording_buffers.pop(camera_id, None)
        self.camera_status.pop(camera_id, None)
        
        # Close WebSocket connections
        str_camera_id = str(camera_id)
        if str_camera_id in self.connected_clients:
            for client in self.connected_clients[str_camera_id]:
                asyncio.create_task(client.close())
            del self.connected_clients[str_camera_id]
        
        self._save_camera_config()
        
        return {"message": f"Camera {camera_id} deleted successfully"}
    
    def start_recording(self, camera_id):
        """Start recording for a camera"""
        if camera_id not in self.cameras:
            raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")
        
        camera = self.cameras[camera_id]
        camera.recording = True
        self._save_camera_config()
        
        return {"message": f"Recording started for camera {camera_id}"}
    
    def stop_recording(self, camera_id):
        """Stop recording for a camera"""
        if camera_id not in self.cameras:
            raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")
        
        camera = self.cameras[camera_id]
        camera.recording = False
        self._save_camera_config()
        
        return {"message": f"Recording stopped for camera {camera_id}"}
    
    def get_models(self):
        """Get all available detection models"""
        return list(self.available_models.values())
    
    def get_current_model(self):
        """Get the currently loaded model"""
        if self.current_model_id:
            return {
                "id": self.current_model_id,
                "name": self.available_models[self.current_model_id].name
            }
        return None
    
    def get_detections(self, camera_id):
        """Get the latest detections for a camera (backward compatibility)"""
        if camera_id not in self.cameras:
            raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")
        
        return self.camera_detections.get(camera_id, [])
    
    def get_frame_generator(self, camera_id):
        """Get a generator that yields frames as JPEG images (backward compatibility)"""
        if camera_id not in self.cameras:
            raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")
        
        def generate():
            while self.active:
                if camera_id in self.camera_frames:
                    frame_data = self.camera_frames[camera_id]
                    frame = frame_data['frame'].copy()
                    
                    # Draw detections on the frame (backward compatibility)
                    if camera_id in self.camera_detections and self.cameras[camera_id].detection_enabled:
                        for det in self.camera_detections[camera_id]:
                            x1, y1, x2, y2 = map(int, det["bbox"])
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, f"{det['class_name']}: {det['confidence']:.2f}", 
                                      (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Encode frame as JPEG
                    _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
                else:
                    # If no frame is available, return a black frame
                    black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    _, jpeg = cv2.imencode('.jpg', black_frame)
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
                
                time.sleep(0.033)  # ~30 FPS
        
        return generate
    
    def cleanup(self):
        """Clean up resources before shutting down"""
        logger.info("Starting cleanup...")
        self.active = False
        
        # Stop all camera threads
        for camera_id, camera_thread in self.camera_threads.items():
            camera_thread.stop()
        
        # Stop processors
        self.single_model_processor.stop()
        self.multi_model_processor.stop()
        
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

app = FastAPI(title="Multi-Camera Multi-Model YOLO Detection API", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===============================
# ORIGINAL API ROUTES (UNCHANGED)
# ===============================

@app.get("/api/cameras", response_model=List[Dict])
async def get_cameras():
    """Get all configured cameras"""
    return camera_manager.get_cameras()

@app.get("/api/cameras/{camera_id}")
async def get_camera(camera_id: Union[int, str]):
    """Get a specific camera by ID"""
    return camera_manager.get_camera(camera_id)

@app.get("/api/cameras/{camera_id}/test")
async def test_camera(camera_id: Union[int, str]):
    """Test camera feed"""
    if camera_id not in camera_manager.camera_frames:
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found or not active")
    
    frame_data = camera_manager.camera_frames[camera_id]
    frame = frame_data['frame']
    _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return Response(content=jpeg.tobytes(), media_type="image/jpeg")

@app.post("/api/cameras")
async def add_camera(camera: dict):
    """Add a new camera"""
    return camera_manager.add_camera(camera)

@app.put("/api/cameras/{camera_id}")
async def update_camera(camera_id: Union[int, str], camera: dict):
    """Update an existing camera"""
    return camera_manager.update_camera(camera_id, camera)

@app.delete("/api/cameras/{camera_id}")
async def delete_camera(camera_id: Union[int, str]):
    """Delete a camera"""
    return camera_manager.delete_camera(camera_id)

@app.post("/api/cameras/{camera_id}/recording/start")
async def start_recording(camera_id: Union[int, str]):
    """Start recording for a camera"""
    return camera_manager.start_recording(camera_id)

@app.post("/api/cameras/{camera_id}/recording/stop")
async def stop_recording(camera_id: Union[int, str]):
    """Stop recording for a camera"""
    return camera_manager.stop_recording(camera_id)

@app.get("/api/models", response_model=List[DetectionModel])
async def get_models():
    """Get all available detection models"""
    return camera_manager.get_models()

@app.get("/api/models/current")
async def get_current_model():
    """Get the currently loaded model"""
    model = camera_manager.get_current_model()
    if not model:
        return {"message": "No model currently loaded"}
    return model

@app.post("/api/models/{model_id}/load")
async def load_model(model_id: str):
    """Load a YOLO model for detection"""
    return await camera_manager.load_model(model_id)

@app.get("/api/cameras/{camera_id}/detections")
async def get_detections(camera_id: Union[int, str]):
    """Get the latest detections for a camera"""
    return camera_manager.get_detections(camera_id)

@app.get("/api/health")
async def health_check():
    return {
        "status": "ok",
        "active_cameras": len([s for s in camera_manager.camera_status.values() if s == "connected"]),
        "total_cameras": len(camera_manager.cameras),
        "model_loaded": camera_manager.current_model_id is not None,
        "multi_model_loaded": len(camera_manager.multi_model_processor.loaded_models)
    }

@app.get("/api/cameras/{camera_id}/stream")
async def stream_camera(camera_id: Union[int, str]):
    """Stream video from a camera as multipart/x-mixed-replace"""
    return StreamingResponse(
        camera_manager.get_frame_generator(camera_id)(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate",
                 "Pragma": "no-cache",
                 "Expires": "0"}
    )

@app.post("/api/cameras/{camera_id}/start_tracking")
async def start_tracking(camera_id: Union[int, str], model_data: dict):
    """Enable object detection for a specific camera"""
    model_id = model_data.get("model")
    
    if not model_id:
        raise HTTPException(status_code=400, detail="Model ID is required")

    if camera_id not in camera_manager.cameras:
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")

    # Load the model if not already loaded
    if camera_manager.current_model_id != model_id:
        await camera_manager.load_model(model_id)

    camera_manager.cameras[camera_id].detection_enabled = True
    return {"message": f"Tracking started for camera {camera_id} using model {model_id}"}

@app.post("/api/cameras/{camera_id}/stop_tracking")
async def stop_tracking(camera_id: Union[int, str]):
    """Disable object detection for a specific camera"""
    if camera_id not in camera_manager.cameras:
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")

    camera_manager.cameras[camera_id].detection_enabled = False
    return {"message": f"Tracking stopped for camera {camera_id}"}

@app.get("/api/cameras/{camera_id}/detection_stream")
async def stream_detection_video(camera_id: Union[int, str]):
    """Stream processed detection frames with bounding boxes"""
    if camera_id not in camera_manager.cameras:
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")

    def generate():
        while camera_manager.active:
            if camera_id in camera_manager.camera_frames:
                frame_data = camera_manager.camera_frames[camera_id]
                frame = frame_data['frame'].copy()

                if camera_manager.cameras[camera_id].detection_enabled:
                    for det in camera_manager.camera_detections.get(camera_id, []):
                        x1, y1, x2, y2 = map(int, det["bbox"])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"{det['class_name']} {det['confidence']:.2f}",
                                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

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

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/api/cameras/{camera_id}/detection_status")
async def get_detection_status(camera_id: Union[int, str]):
    """Get real-time detection status for a camera"""
    if camera_id not in camera_manager.cameras:
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")
    
    camera = camera_manager.cameras[camera_id]
    detections = camera_manager.camera_detections.get(camera_id, [])
    
    # Get recent detection statistics
    recent_detections = [d for d in detections if 
                        (datetime.now() - datetime.fromisoformat(d["timestamp"])).seconds < 60]
    
    return {
        "camera_id": camera_id,
        "detection_enabled": camera.detection_enabled,
        "camera_status": camera_manager.camera_status.get(camera_id, "disconnected"),
        "model_loaded": camera_manager.current_model_id is not None,
        "current_model": camera_manager.current_model_id,
        "alert_threshold": camera.alert_threshold,
        "total_detections": len(detections),
        "recent_detections": len(recent_detections),
        "last_detection": detections[0] if detections else None,
        "websocket_clients": len(camera_manager.connected_clients.get(str(camera_id), []))
    }

@app.websocket("/ws/cameras/{camera_id}")
async def websocket_camera_alerts(websocket: WebSocket, camera_id: Union[int, str]):
    """Enhanced WebSocket endpoint for camera alerts"""
    await websocket.accept()
    
    str_camera_id = str(camera_id)
    
    # Send initial connection confirmation
    await websocket.send_json({
        "type": "connection",
        "status": "connected",
        "camera_id": str_camera_id,
        "message": f"Connected to alerts for camera {camera_id}",
        "timestamp": datetime.now().isoformat()
    })
    
    # Register this client
    await camera_manager.register_client(camera_id, websocket)
    
    try:
        # Keep the connection alive and handle incoming messages
        while True:
            try:
                # Wait for messages with timeout
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                
                try:
                    message = json.loads(data)
                    message_type = message.get("type")
                    
                    if message_type == "ping":
                        # Respond to ping with pong
                        await websocket.send_json({
                            "type": "pong",
                            "timestamp": datetime.now().isoformat()
                        })
                    elif message_type == "get_status":
                        # Send current detection status
                        status = await get_detection_status(camera_id)
                        await websocket.send_json({
                            "type": "status",
                            "data": status
                        })
                    elif message_type == "subscribe_alerts":
                        # Confirmation that client wants to receive alerts
                        await websocket.send_json({
                            "type": "subscription_confirmed",
                            "camera_id": str_camera_id,
                            "message": "Successfully subscribed to alerts"
                        })
                        
                except json.JSONDecodeError:
                    # Invalid JSON, send error response
                    await websocket.send_json({
                        "type": "error",
                        "message": "Invalid JSON format"
                    })
                    
            except asyncio.TimeoutError:
                # Send heartbeat ping
                await websocket.send_json({
                    "type": "heartbeat",
                    "timestamp": datetime.now().isoformat()
                })
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected from camera {camera_id}")
    except Exception as e:
        logger.error(f"WebSocket error for camera {camera_id}: {e}")
    finally:
        # Always clean up
        await camera_manager.unregister_client(camera_id, websocket)

@app.get("/api/detection/health")
async def detection_health():
    """Get overall detection system health"""
    active_cameras = sum(1 for status in camera_manager.camera_status.values() 
                        if status == "connected")
    
    detection_enabled_cameras = sum(1 for cam in camera_manager.cameras.values() 
                                   if cam.detection_enabled)
    
    total_connections = sum(len(clients) for clients in camera_manager.connected_clients.values())
    
    return {
        "model_loaded": camera_manager.current_model_id is not None,
        "current_model": camera_manager.current_model_id,
        "total_cameras": len(camera_manager.cameras),
        "active_cameras": active_cameras,
        "detection_enabled_cameras": detection_enabled_cameras,
        "total_websocket_connections": total_connections,
        "single_model_processor_active": camera_manager.single_model_processor.active,
        "multi_model_processor_active": camera_manager.multi_model_processor.active,
        "loaded_preview_models": list(camera_manager.multi_model_processor.loaded_models.keys()),
        "system_status": "healthy" if camera_manager.current_model_id else "no_model_loaded"
    }

# ===============================
# NEW MULTI-MODEL API ROUTES
# ===============================

@app.post("/api/cameras/{camera_id}/models/{model_id}/load_preview")
async def load_model_for_preview(camera_id: Union[int, str], model_id: str, subscriber_data: dict = {}):
    """Load a model for preview on a specific camera"""
    if camera_id not in camera_manager.cameras:
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")
    
    subscriber_id = subscriber_data.get("subscriber_id", f"{camera_id}_{model_id}_{int(time.time())}")
    
    result = await camera_manager.load_detection_model_for_preview(model_id, subscriber_id)
    result["subscriber_id"] = subscriber_id
    
    return result

@app.delete("/api/cameras/{camera_id}/models/{model_id}/unload_preview")
async def unload_model_from_preview(camera_id: Union[int, str], model_id: str, subscriber_data: dict = {}):
    """Unload a model from preview on a specific camera"""
    if camera_id not in camera_manager.cameras:
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")
    
    subscriber_id = subscriber_data.get("subscriber_id")
    return camera_manager.unload_detection_model_for_preview(model_id, subscriber_id)

@app.get("/api/cameras/{camera_id}/models/{model_id}/stream")
async def stream_camera_with_model(camera_id: Union[int, str], model_id: str):
    """Stream camera feed with detections from a specific model"""
    if camera_id not in camera_manager.cameras:
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")
    
    if model_id not in camera_manager.available_models:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    
    # Ensure model is loaded for preview
    if model_id not in camera_manager.multi_model_processor.loaded_models:
        await camera_manager.load_detection_model_for_preview(model_id)
    
    return StreamingResponse(
        camera_manager.get_model_detection_frame_generator(camera_id, model_id)(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )

@app.get("/api/cameras/{camera_id}/models/{model_id}/detections")
async def get_model_detections(camera_id: Union[int, str], model_id: str):
    """Get detections for a specific camera-model combination"""
    if camera_id not in camera_manager.cameras:
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")
    
    if model_id not in camera_manager.available_models:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    
    detections = camera_manager.get_multi_model_detections(camera_id, model_id)
    
    return {
        "camera_id": camera_id,
        "model_id": model_id,
        "detections": detections,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/cameras/{camera_id}/multi_model_detections")
async def get_all_model_detections_for_camera(camera_id: Union[int, str]):
    """Get detections from all loaded models for a specific camera"""
    if camera_id not in camera_manager.cameras:
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")
    
    all_detections = camera_manager.get_multi_model_detections(camera_id)
    
    return {
        "camera_id": camera_id,
        "models": all_detections,
        "timestamp": datetime.now().isoformat(),
        "loaded_models": list(camera_manager.multi_model_processor.loaded_models.keys())
    }

@app.get("/api/cameras/{camera_id}/models/{model_id}/compare")
async def compare_model_detections(camera_id: Union[int, str], model_id: str, compare_with: str):
    """Compare detections between two models for the same camera"""
    if camera_id not in camera_manager.cameras:
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")
    
    if model_id not in camera_manager.available_models:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    
    if compare_with not in camera_manager.available_models:
        raise HTTPException(status_code=404, detail=f"Compare model {compare_with} not found")
    
    detections_a = camera_manager.get_multi_model_detections(camera_id, model_id)
    detections_b = camera_manager.get_multi_model_detections(camera_id, compare_with)
    
    # Basic comparison metrics
    comparison = {
        "camera_id": camera_id,
        "model_a": {
            "id": model_id,
            "detections": detections_a,
            "count": len(detections_a)
        },
        "model_b": {
            "id": compare_with,
            "detections": detections_b,
            "count": len(detections_b)
        },
        "comparison": {
            "detection_count_diff": len(detections_a) - len(detections_b),
            "unique_to_a": [],
            "unique_to_b": [],
            "common": []
        },
        "timestamp": datetime.now().isoformat()
    }
    
    # Simple comparison based on class names and approximate bbox overlap
    classes_a = {det["class_name"] for det in detections_a}
    classes_b = {det["class_name"] for det in detections_b}
    
    comparison["comparison"]["unique_classes_a"] = list(classes_a - classes_b)
    comparison["comparison"]["unique_classes_b"] = list(classes_b - classes_a)
    comparison["comparison"]["common_classes"] = list(classes_a & classes_b)
    
    return comparison

@app.get("/api/multi_model/status")
async def get_multi_model_status():
    """Get status of all loaded models for preview"""
    loaded_models = camera_manager.multi_model_processor.loaded_models
    model_subscribers = camera_manager.multi_model_processor.model_subscribers
    
    status = {
        "loaded_models": [],
        "total_loaded": len(loaded_models),
        "active_subscribers": 0,
        "frame_queue_status": {},
        "timestamp": datetime.now().isoformat()
    }
    
    for model_id, model in loaded_models.items():
        subscribers = len(model_subscribers.get(model_id, set()))
        queue_size = camera_manager.multi_model_processor.frame_queues[model_id].qsize() if model_id in camera_manager.multi_model_processor.frame_queues else 0
        
        model_info = {
            "model_id": model_id,
            "model_name": camera_manager.available_models[model_id].name,
            "subscribers": subscribers,
            "queue_size": queue_size,
            "active": model_id in camera_manager.multi_model_processor.model_threads
        }
        
        status["loaded_models"].append(model_info)
        status["active_subscribers"] += subscribers
        status["frame_queue_status"][model_id] = queue_size
    
    return status

@app.post("/api/cameras/{camera_id}/models/bulk_load")
async def bulk_load_models_for_preview(camera_id: Union[int, str], models_data: dict):
    """Load multiple models for preview on a specific camera"""
    if camera_id not in camera_manager.cameras:
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")
    
    model_ids = models_data.get("model_ids", [])
    subscriber_id = models_data.get("subscriber_id", f"{camera_id}_bulk_{int(time.time())}")
    
    results = []
    for model_id in model_ids:
        try:
            result = await camera_manager.load_detection_model_for_preview(model_id, subscriber_id)
            results.append({
                "model_id": model_id,
                "success": result.get("success", False),
                "message": result.get("message", "")
            })
        except Exception as e:
            results.append({
                "model_id": model_id,
                "success": False,
                "message": str(e)
            })
    
    return {
        "camera_id": camera_id,
        "subscriber_id": subscriber_id,
        "results": results,
        "loaded_models": [r["model_id"] for r in results if r["success"]],
        "timestamp": datetime.now().isoformat()
    }

@app.delete("/api/cameras/{camera_id}/models/bulk_unload")
async def bulk_unload_models_from_preview(camera_id: Union[int, str], models_data: dict):
    """Unload multiple models from preview on a specific camera"""
    if camera_id not in camera_manager.cameras:
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")
    
    model_ids = models_data.get("model_ids", [])
    subscriber_id = models_data.get("subscriber_id")
    
    results = []
    for model_id in model_ids:
        try:
            result = camera_manager.unload_detection_model_for_preview(model_id, subscriber_id)
            results.append({
                "model_id": model_id,
                "success": True,
                "message": result.get("message", "")
            })
        except Exception as e:
            results.append({
                "model_id": model_id,
                "success": False,
                "message": str(e)
            })
    
    return {
        "camera_id": camera_id,
        "results": results,
        "unloaded_models": [r["model_id"] for r in results if r["success"]],
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/cameras/{camera_id}/models/grid_stream")
async def stream_multi_model_grid(camera_id: Union[int, str], model_ids: str = ""):
    """Stream a grid view of multiple model detections for the same camera"""
    if camera_id not in camera_manager.cameras:
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")
    
    # Parse model IDs from query parameter
    if model_ids:
        model_list = [mid.strip() for mid in model_ids.split(",")]
    else:
        model_list = list(camera_manager.multi_model_processor.loaded_models.keys())
    
    # Ensure models are loaded
    for model_id in model_list:
        if model_id not in camera_manager.multi_model_processor.loaded_models:
            await camera_manager.load_detection_model_for_preview(model_id)
    
    def generate_grid():
        while camera_manager.active:
            if camera_id in camera_manager.camera_frames:
                frame_data = camera_manager.camera_frames[camera_id]
                original_frame = frame_data['frame']
                
                # Create grid based on number of models
                num_models = len(model_list)
                if num_models == 0:
                    # No models, return original frame
                    _, jpeg = cv2.imencode('.jpg', original_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
                    time.sleep(0.033)
                    continue
                
                # Calculate grid dimensions
                cols = min(2, num_models)  # Max 2 columns
                rows = (num_models + cols - 1) // cols
                
                # Resize frames for grid
                frame_height, frame_width = original_frame.shape[:2]
                grid_frame_width = frame_width // cols
                grid_frame_height = frame_height // rows
                
                # Create grid canvas
                grid_height = grid_frame_height * rows
                grid_width = grid_frame_width * cols
                grid_frame = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
                
                # Fill grid with model detection frames
                for idx, model_id in enumerate(model_list):
                    row = idx // cols
                    col = idx % cols
                    
                    # Get detections for this model
                    detections = camera_manager.multi_model_processor.get_detections(camera_id, model_id)
                    
                    # Create frame with detections
                    model_frame = original_frame.copy()
                    color = camera_manager._get_model_color(model_id)
                    
                    for det in detections:
                        x1, y1, x2, y2 = map(int, det["bbox"])
                        cv2.rectangle(model_frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(model_frame, f"{det['class_name']}: {det['confidence']:.2f}", 
                                  (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # Add model name overlay
                    cv2.putText(model_frame, f"Model: {model_id}", (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    
                    # Resize and place in grid
                    resized_frame = cv2.resize(model_frame, (grid_frame_width, grid_frame_height))
                    
                    y_start = row * grid_frame_height
                    y_end = y_start + grid_frame_height
                    x_start = col * grid_frame_width
                    x_end = x_start + grid_frame_width
                    
                    grid_frame[y_start:y_end, x_start:x_end] = resized_frame
                
                # Encode and yield grid frame
                _, jpeg = cv2.imencode('.jpg', grid_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            else:
                # Return black frame if camera not available
                black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                _, jpeg = cv2.imencode('.jpg', black_frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            
            time.sleep(0.033)  # ~30 FPS
    
    return StreamingResponse(
        generate_grid(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )

# ===============================
# ENHANCED WEBSOCKET FOR MULTI-MODEL
# ===============================

@app.websocket("/ws/cameras/{camera_id}/multi_model")
async def websocket_multi_model_updates(websocket: WebSocket, camera_id: Union[int, str]):
    """WebSocket endpoint for multi-model detection updates"""
    await websocket.accept()
    
    str_camera_id = str(camera_id)
    
    # Send initial connection confirmation
    await websocket.send_json({
        "type": "connection",
        "status": "connected",
        "camera_id": str_camera_id,
        "message": f"Connected to multi-model updates for camera {camera_id}",
        "available_models": list(camera_manager.available_models.keys()),
        "loaded_models": list(camera_manager.multi_model_processor.loaded_models.keys()),
        "timestamp": datetime.now().isoformat()
    })
    
    subscribed_models = set()
    
    try:
        while True:
            try:
                # Wait for messages with timeout
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                
                try:
                    message = json.loads(data)
                    message_type = message.get("type")
                    
                    if message_type == "ping":
                        await websocket.send_json({
                            "type": "pong",
                            "timestamp": datetime.now().isoformat()
                        })
                        
                    elif message_type == "subscribe_model":
                        model_id = message.get("model_id")
                        if model_id and model_id in camera_manager.available_models:
                            # Load model if not already loaded
                            if model_id not in camera_manager.multi_model_processor.loaded_models:
                                await camera_manager.load_detection_model_for_preview(model_id)
                            
                            subscribed_models.add(model_id)
                            await websocket.send_json({
                                "type": "subscription_confirmed",
                                "model_id": model_id,
                                "message": f"Subscribed to model {model_id}"
                            })
                        else:
                            await websocket.send_json({
                                "type": "error",
                                "message": f"Invalid model ID: {model_id}"
                            })
                            
                    elif message_type == "unsubscribe_model":
                        model_id = message.get("model_id")
                        subscribed_models.discard(model_id)
                        await websocket.send_json({
                            "type": "unsubscription_confirmed",
                            "model_id": model_id,
                            "message": f"Unsubscribed from model {model_id}"
                        })
                        
                    elif message_type == "get_detections":
                        detections_data = {}
                        for model_id in subscribed_models:
                            detections = camera_manager.get_multi_model_detections(camera_id, model_id)
                            detections_data[model_id] = detections
                        
                        await websocket.send_json({
                            "type": "detections_update",
                            "camera_id": str_camera_id,
                            "detections": detections_data,
                            "timestamp": datetime.now().isoformat()
                        })
                        
                    elif message_type == "get_status":
                        status = {
                            "camera_id": str_camera_id,
                            "camera_status": camera_manager.camera_status.get(camera_id, "unknown"),
                            "subscribed_models": list(subscribed_models),
                            "loaded_models": list(camera_manager.multi_model_processor.loaded_models.keys()),
                            "available_models": list(camera_manager.available_models.keys())
                        }
                        await websocket.send_json({
                            "type": "status_update",
                            "data": status
                        })
                        
                except json.JSONDecodeError:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Invalid JSON format"
                    })
                    
            except asyncio.TimeoutError:
                # Send heartbeat with current detection counts
                if subscribed_models:
                    detection_counts = {}
                    for model_id in subscribed_models:
                        detections = camera_manager.get_multi_model_detections(camera_id, model_id)
                        detection_counts[model_id] = len(detections)
                    
                    await websocket.send_json({
                        "type": "heartbeat",
                        "detection_counts": detection_counts,
                        "timestamp": datetime.now().isoformat()
                    })
                else:
                    await websocket.send_json({
                        "type": "heartbeat",
                        "timestamp": datetime.now().isoformat()
                    })
                
    except WebSocketDisconnect:
        logger.info(f"Multi-model WebSocket client disconnected from camera {camera_id}")
    except Exception as e:
        logger.error(f"Multi-model WebSocket error for camera {camera_id}: {e}")
    finally:
        # Clean up subscriptions
        for model_id in subscribed_models:
            camera_manager.multi_model_processor.unsubscribe_from_model(model_id, f"ws_{camera_id}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)