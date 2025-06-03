from typing import Dict, List, Optional
import threading
from datetime import datetime
import cv2
import numpy as np
from ultralytics import YOLO

from config.settings import DEFAULT_DETECTION_CONF, DEFAULT_MODELS
from models.detection import DetectionModel, Detection


class DetectionService:
    """Service for managing object detection"""
    
    def __init__(self):
        self.detection_model = None
        self.current_model_id = None
        self.available_models = self._load_available_models()
        
    def _load_available_models(self) -> Dict[str, DetectionModel]:
        """Load available detection models"""
        models = {}
        
        # Default models from ultralytics
        for model_info in DEFAULT_MODELS:
            models[model_info["id"]] = DetectionModel(**model_info)
            
        # You could scan the models directory for custom models here
        return models
    
    def load_model(self, model_id: str) -> bool:
        """Load a YOLO model for detection"""
        if model_id not in self.available_models:
            return False
        
        model = self.available_models[model_id]
        
        try:
            self.detection_model = YOLO(model.path)
            self.current_model_id = model_id
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def get_current_model(self) -> Optional[Dict]:
        """Get the currently loaded model"""
        if self.current_model_id:
            return {
                "id": self.current_model_id,
                "name": self.available_models[self.current_model_id].name
            }
        return None
    
    def get_models(self) -> List[DetectionModel]:
        """Get all available detection models"""
        return list(self.available_models.values())
    
    def detect_objects(self, frame: np.ndarray, confidence: float = DEFAULT_DETECTION_CONF) -> List[Detection]:
        """Perform object detection on a frame"""
        if self.detection_model is None:
            return []
        
        results = self.detection_model.predict(frame, conf=confidence)
        detections = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = box.conf.item()
                class_id = int(box.cls.item())
                class_name = result.names[class_id]
                
                detection = Detection(
                    bbox=[x1, y1, x2, y2],
                    confidence=confidence,
                    class_id=class_id,
                    class_name=class_name,
                    timestamp=datetime.now().isoformat()
                )
                detections.append(detection)
        
        return detections


# Create a global instance
detection_service = DetectionService()