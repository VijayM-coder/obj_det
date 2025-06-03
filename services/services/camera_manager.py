import cv2
import numpy as np
import threading
import time
import json
import uuid
import asyncio
import base64
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from fastapi import WebSocket, HTTPException

from config.settings import (
    CAMERA_CONFIG_PATH, 
    PRE_RECORDING_BUFFER_SECONDS, 
    DEFAULT_JPEG_QUALITY
)
from models.camera import Camera, CameraConfig
from models.alert import Alert
from models.detection import Detection
from utils.video_utils import (
    encode_frame_jpeg, 
    draw_detections, 
    create_mjpeg_frame,
    parse_resolution
)
from services.detection_service import detection_service
from services.recording_service import recording_service


class CameraManager:
    """Service for managing camera operations"""
    
    def __init__(self):
        self.cameras: Dict[Union[int, str], Camera] = {}  # Camera configs
        self.camera_streams: Dict[Union[int, str], cv2.VideoCapture] = {}  # OpenCV capture objects
        self.camera_frames: Dict[Union[int, str], np.ndarray] = {}  # Latest frames
        self.camera_detections: Dict[Union[int, str], List] = {}  # Latest detections
        self.camera_status: Dict[Union[int, str], str] = {}  # Camera connection status
        self.pre_recording_buffers: Dict[Union[int, str], List] = {}  # Pre-recording buffers
        self.active = True
        self.connected_clients: Dict[str, List[WebSocket]] = {}  # WebSocket clients per camera
        self.loop = asyncio.get_event_loop()  # Get the event loop
        
        # Load camera configurations
        self._load_camera_config()
        
        # Start camera threads
        self._start_camera_threads()
    
    def _load_camera_config(self):
        """Load camera configurations from file"""
        try:
            with open(CAMERA_CONFIG_PATH, 'r') as f:
                config_data = json.load(f)
                
            # Check if the config is in the new format (with "cameras" key)
            if isinstance(config_data, dict) and "cameras" in config_data:
                camera_configs = config_data["cameras"]
            else:
                # Old format or just a list of cameras
                camera_configs = config_data
                
            for camera_config in camera_configs:
                camera_id = camera_config.get("id")
                self.cameras[camera_id] = Camera(**camera_config)
                self.camera_detections[camera_id] = []
                self.pre_recording_buffers[camera_id] = []
                self.camera_status[camera_id] = "disconnected"
                
                # Initialize the WebSocket clients list for this camera
                self.connected_clients[str(camera_id)] = []
                
        except FileNotFoundError:
            print(f"Camera configuration file not found at {CAMERA_CONFIG_PATH}")
            # Creating an empty config file
            with open(CAMERA_CONFIG_PATH, 'w') as f:
                json.dump({"cameras": []}, f)
        except json.JSONDecodeError:
            print(f"Invalid JSON in camera configuration file")
            raise
    
    def _save_camera_config(self):
        """Save camera configurations to file"""
        with open(CAMERA_CONFIG_PATH, 'w') as f:
            config = {"cameras": [cam.model_dump() for cam in self.cameras.values()]}
            json.dump(config, f, indent=2)
    
    def _start_camera_threads(self):
        """Start processing threads for all enabled cameras"""
        for camera_id, camera in self.cameras.items():
            if camera.enabled:
                thread = threading.Thread(target=self._camera_processing_thread, args=(camera_id,))
                thread.daemon = True
                thread.start()
    
    def _get_camera_source(self, camera):
        """Convert the camera source to the proper format for OpenCV"""
        if camera.type == "webcam":
            # For webcams, convert the index to int
            return int(camera.source)
        else:
            # For IP cameras, use the URL string
            return camera.source
    
    def _camera_processing_thread(self, camera_id):
        """Thread to handle camera capture, processing, and recording"""
        camera = self.cameras[camera_id]
        source = self._get_camera_source(camera)
        
        cap = cv2.VideoCapture(source)
        
        # Try to set the resolution and FPS if specified
        if camera.resolution:
            try:
                width, height = parse_resolution(camera.resolution)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            except ValueError:
                print(f"Invalid resolution format for camera {camera.name}: {camera.resolution}")
        
        if camera.fps:
            cap.set(cv2.CAP_PROP_FPS, camera.fps)
        
        if not cap.isOpened():
            print(f"Failed to open camera: {camera.name} ({source})")
            self.camera_status[camera_id] = "error"
            return
            
        self.camera_streams[camera_id] = cap
        self.camera_status[camera_id] = "connected"
        
        # Get actual FPS (may differ from requested)
        fps = cap.get(cv2.CAP_PROP_FPS)
        pre_recording_frames_count = int(fps * PRE_RECORDING_BUFFER_SECONDS)
        
        while self.active:
            success, frame = cap.read()
            if not success:
                print(f"Failed to read frame from camera {camera.name}")
                self.camera_status[camera_id] = "disconnected"
                time.sleep(1)  # Wait before retry
                # Try to reconnect
                cap.release()
                cap = cv2.VideoCapture(source)
                continue
                
            # Update status to connected
            self.camera_status[camera_id] = "connected"
                
            # Store the current frame
            self.camera_frames[camera_id] = frame.copy()
            
            # Update pre-recording buffer
            self.pre_recording_buffers[camera_id].append(frame.copy())
            if len(self.pre_recording_buffers[camera_id]) > pre_recording_frames_count:
                self.pre_recording_buffers[camera_id].pop(0)
            
            # Process detections if model is loaded and detection is enabled
            if detection_service.detection_model and camera.detection_enabled:
                detections = detection_service.detect_objects(frame, camera_id)
                self.camera_detections[camera_id] = detections
                
                # Check for high-confidence detections and send alerts
                for detection in detections:
                    if detection["confidence"] > camera.alert_threshold:
                        self._send_alert(camera_id, detection, frame)
            
            # Handle recording
            if camera.recording:
                # Start recording if not already recording
                if not recording_service.is_recording(camera_id):
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    recording_service.start_recording(
                        camera_id, width, height, fps, 
                        self.pre_recording_buffers[camera_id]
                    )
                
                # Write current frame
                recording_service.write_frame(camera_id, frame)
            elif recording_service.is_recording(camera_id):
                # Stop recording if recording flag was turned off
                recording_service.stop_recording(camera_id)
            
            # Broadcast frame to connected clients
            self._broadcast_frame(camera_id, frame)
            
            # A small delay to reduce CPU usage
            time.sleep(0.01)
        
        # Clean up resources
        if recording_service.is_recording(camera_id):
            recording_service.stop_recording(camera_id)
        cap.release()
        self.camera_status[camera_id] = "disconnected"
    
    def _send_alert(self, camera_id, detection, frame):
        """Send an alert for high-confidence detections"""
        img_base64 = base64.b64encode(encode_frame_jpeg(frame)).decode('utf-8')
        
        alert = Alert(
            camera_id=camera_id,
            timestamp=detection["timestamp"],
            object_type=detection["class_name"],
            confidence=detection["confidence"],
            bbox=detection["bbox"],
            image_data=img_base64
        )
        
        print(f"ALERT: {alert.object_type} detected on camera {camera_id} with confidence {alert.confidence:.2f}")
        
        # Broadcast alert to WebSocket clients
        self._broadcast_alert(camera_id, alert)
    
    async def _broadcast_alert_async(self, camera_id, alert):
        """Broadcast alert to WebSocket clients asynchronously"""
        str_camera_id = str(camera_id)
        if str_camera_id in self.connected_clients:
            disconnected_clients = []
            for client in self.connected_clients[str_camera_id]:
                try:
                    await client.send_json({
                        "type": "alert",
                        "camera_id": str(camera_id),
                        "alert": alert.model_dump(exclude={"image_data"})
                    })
                except Exception:
                    disconnected_clients.append(client)
            
            # Remove disconnected clients
            for client in disconnected_clients:
                if client in self.connected_clients[str_camera_id]:
                    self.connected_clients[str_camera_id].remove(client)
    
    def _broadcast_alert(self, camera_id, alert):
        """Create a task to broadcast alert asynchronously"""
        asyncio.run_coroutine_threadsafe(self._broadcast_alert_async(camera_id, alert), self.loop)
    
    async def _broadcast_frame_async(self, camera_id, frame):
        """Broadcast frame to WebSocket clients asynchronously"""
        str_camera_id = str(camera_id)
        if str_camera_id in self.connected_clients and self.connected_clients[str_camera_id]:
            # Encode frame as JPEG
            jpeg_data = encode_frame_jpeg(frame, DEFAULT_JPEG_QUALITY)
            
            disconnected_clients = []
            for client in self.connected_clients[str_camera_id]:
                try:
                    await client.send_bytes(jpeg_data)
                except Exception:
                    disconnected_clients.append(client)
            
            # Remove disconnected clients
            for client in disconnected_clients:
                if client in self.connected_clients[str_camera_id]:
                    self.connected_clients[str_camera_id].remove(client)
    
    def _broadcast_frame(self, camera_id, frame):
        """Create a task to broadcast frame asynchronously"""
        asyncio.run_coroutine_threadsafe(self._broadcast_frame_async(camera_id, frame), self.loop)
    
    async def register_client(self, camera_id, websocket):
        """Register a WebSocket client for a camera"""
        str_camera_id = str(camera_id)
        if str_camera_id not in self.connected_clients:
            self.connected_clients[str_camera_id] = []
        self.connected_clients[str_camera_id].append(websocket)
    
    async def unregister_client(self, camera_id, websocket):
        """Unregister a WebSocket client"""
        str_camera_id = str(camera_id)
        if str_camera_id in self.connected_clients and websocket in self.connected_clients[str_camera_id]:
            self.connected_clients[str_camera_id].remove(websocket)
    
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
            thread = threading.Thread(target=self._camera_processing_thread, args=(camera_id,))
            thread.daemon = True
            thread.start()
        
        # Save the updated configuration
        self._save_camera_config()
        
        # Return with status
        camera_dict = camera.model_dump()
        camera_dict["status"] = self.camera_status.get(camera_id, "disconnected")
        return camera_dict
    
    def update_camera(self, camera_id, camera_data):
        """Update an existing camera"""
        if camera_id not in self.cameras:
            raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")
        
        # Update camera attributes
        current_camera = self.cameras[camera_id]
        was_enabled = current_camera.enabled
        
        # Ensure ID remains the same
        camera_data["id"] = camera_id
        
        updated_camera = Camera(**camera_data)
        self.cameras[camera_id] = updated_camera
        
        # Handle changes to enabled status
        if not was_enabled and updated_camera.enabled:
            # Camera was turned on
            thread = threading.Thread(target=self._camera_processing_thread, args=(camera_id,))
            thread.daemon = True
            thread.start()
        elif was_enabled and not updated_camera.enabled:
            # Camera was turned off
            if camera_id in self.camera_streams:
                self.camera_streams[camera_id].release()
                self.camera_status[camera_id] = "disconnected"
        
        # Save the updated configuration
        self._save_camera_config()
        
        # Return with status
        camera_dict = updated_camera.model_dump()
        camera_dict["status"] = self.camera_status.get(camera_id, "disconnected")
        return camera_dict
    
    def delete_camera(self, camera_id):
        """Delete a camera"""
        if camera_id not in self.cameras:
            raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")
        
        # Stop camera processing if running
        if camera_id in self.camera_streams:
            self.camera_streams[camera_id].release()
            del self.camera_streams[camera_id]
        
        # Remove camera from dictionaries
        del self.cameras[camera_id]
        if camera_id in self.camera_detections:
            del self.camera_detections[camera_id]
        if camera_id in self.pre_recording_buffers:
            del self.pre_recording_buffers[camera_id]
        if camera_id in self.camera_status:
            del self.camera_status[camera_id]
        
        # Close all WebSocket connections for this camera
        str_camera_id = str(camera_id)
        if str_camera_id in self.connected_clients:
            for client in self.connected_clients[str_camera_id]:
                asyncio.create_task(client.close())
            del self.connected_clients[str_camera_id]
        
        # Save the updated configuration
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
    
    def get_frame_generator(self, camera_id):
        """Get a generator that yields frames as MJPEG frames"""
        if camera_id not in self.cameras:
            raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")
        
        def generate():
            while self.active:
                if camera_id in self.camera_frames:
                    frame = self.camera_frames[camera_id].copy()
                    
                    # Draw detections on the frame if enabled
                    if detection_service.detection_model and self.cameras[camera_id].detection_enabled:
                        frame = draw_detections(frame, self.camera_detections.get(camera_id, []))
                    
                    # Create MJPEG frame
                    yield create_mjpeg_frame(frame, DEFAULT_JPEG_QUALITY)
                else:
                    # If no frame is available, return a black frame
                    black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    yield create_mjpeg_frame(black_frame, DEFAULT_JPEG_QUALITY)
                
                time.sleep(0.033)  # ~30 FPS
        
        return generate
    
    def cleanup(self):
        """Clean up resources before shutting down"""
        self.active = False
        time.sleep(0.5)  # Give threads time to finish
        
        # Release all camera streams
        for camera_id, cap in self.camera_streams.items():
            if cap:
                cap.release()
        
        # Stop all recordings
        for camera_id in self.cameras.keys():
            if recording_service.is_recording(camera_id):
                recording_service.stop_recording(camera_id)