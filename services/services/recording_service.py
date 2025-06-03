import cv2
import numpy as np
from datetime import datetime
import os
from typing import List, Dict, Any, Optional, Union

from config.settings import RECORDINGS_DIR, PRE_RECORDING_BUFFER_SECONDS


class RecordingService:
    """Service for handling video recording"""
    
    def __init__(self):
        self.recording_writers: Dict[Union[int, str], cv2.VideoWriter] = {}
        self.recording_paths: Dict[Union[int, str], str] = {}
        
    def start_recording(self, camera_id: Union[int, str], width: int, height: int, fps: float,
                        pre_buffer: List[np.ndarray] = None) -> str:
        """
        Start recording for a camera
        
        Args:
            camera_id: Camera identifier
            width: Frame width
            height: Frame height
            fps: Frames per second
            pre_buffer: List of frames to include at the start (pre-recording buffer)
            
        Returns:
            Path to the recording file
        """
        # Stop any existing recording
        self.stop_recording(camera_id)
        
        # Create new recording file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        recording_path = os.path.join(RECORDINGS_DIR, f"{camera_id}_{timestamp}.mp4")
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.recording_writers[camera_id] = cv2.VideoWriter(recording_path, fourcc, fps, (width, height))
        self.recording_paths[camera_id] = recording_path
        
        # Write pre-buffer frames if provided
        if pre_buffer:
            for frame in pre_buffer:
                self.recording_writers[camera_id].write(frame)
        
        return recording_path
    
    def write_frame(self, camera_id: Union[int, str], frame: np.ndarray) -> bool:
        """
        Write a frame to the recording
        
        Args:
            camera_id: Camera identifier
            frame: Frame to write
            
        Returns:
            Success status
        """
        if camera_id in self.recording_writers and self.recording_writers[camera_id]:
            self.recording_writers[camera_id].write(frame)
            return True
        return False
    
    def stop_recording(self, camera_id: Union[int, str]) -> Optional[str]:
        """
        Stop recording for a camera
        
        Args:
            camera_id: Camera identifier
            
        Returns:
            Path to the recording file or None if no recording was active
        """
        recording_path = None
        
        if camera_id in self.recording_writers and self.recording_writers[camera_id]:
            self.recording_writers[camera_id].release()
            recording_path = self.recording_paths.get(camera_id)
            self.recording_writers[camera_id] = None
            
        if camera_id in self.recording_paths:
            recording_path = self.recording_paths.pop(camera_id)
            
        return recording_path
    
    def is_recording(self, camera_id: Union[int, str]) -> bool:
        """Check if a camera is currently recording"""
        return camera_id in self.recording_writers and self.recording_writers[camera_id] is not None


# Create a global instance
recording_service = RecordingService()