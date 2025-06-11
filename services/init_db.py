#!/usr/bin/env python3
"""
Database initialization script for the Multi-Camera YOLO Detection System
"""

import os
import sys
from datetime import datetime
import logging

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database import init_database, test_connection, get_db_session
from services import ModelService, CameraService
from schemas import ModelInfoCreate, CameraCreate

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_default_models():
    """Create default YOLO models in the database"""
    default_models = [
        {
            "id": "yolov8n",
            "name": "YOLOv8 Nano",
            "path": "./models/yolov8n.pt",
            "description": "Fastest YOLOv8 model, good for real-time applications",
            "input_size": "640x640",
            "confidence_threshold": 0.25
        },
        {
            "id": "yolov8s",
            "name": "YOLOv8 Small",
            "path": "./models/yolov8s.pt",
            "description": "Balanced speed and accuracy",
            "input_size": "640x640",
            "confidence_threshold": 0.25
        },
        {
            "id": "yolov8m",
            "name": "YOLOv8 Medium",
            "path": "./models/yolov8m.pt",
            "description": "Good accuracy with moderate speed",
            "input_size": "640x640",
            "confidence_threshold": 0.25
        },
        {
            "id": "yolov8l",
            "name": "YOLOv8 Large",
            "path": "./models/yolov8l.pt",
            "description": "High accuracy model",
            "input_size": "640x640",
            "confidence_threshold": 0.25
        },
        {
            "id": "yolov8x",
            "name": "YOLOv8 Extra Large",
            "path": "./models/yolov8x.pt",
            "description": "Highest accuracy YOLOv8 model",
            "input_size": "640x640",
            "confidence_threshold": 0.25
        }
    ]
    
    with get_db_session() as db:
        for model_data in default_models:
            try:
                # Check if model already exists
                existing_model = ModelService.get_model(db, model_data["id"])
                if not existing_model:
                    model_create = ModelInfoCreate(**model_data)
                    ModelService.create_model(db, model_create)
                    logger.info(f"Created model: {model_data['name']}")
                else:
                    logger.info(f"Model already exists: {model_data['name']}")
            except Exception as e:
                logger.error(f"Failed to create model {model_data['name']}: {e}")

def create_sample_cameras():
    """Create sample cameras for testing"""
    sample_cameras = [
        {
            "id": "webcam_0",
            "name": "Default Webcam",
            "type": "webcam",
            "source": "0",
            "resolution": "640x480",
            "fps": 30,
            "enabled": True,
            "detection_enabled": True,
            "alert_threshold": 0.5
        },
        {
            "id": "ip_camera_1",
            "name": "IP Camera Example",
            "type": "ip",
            "source": "rtsp://admin:password@192.168.1.100:554/stream",
            "resolution": "1920x1080",
            "fps": 25,
            "enabled": False,  # Disabled by default since it's an example
            "detection_enabled": True,
            "alert_threshold": 0.6
        }
    ]
    
    with get_db_session() as db:
        for camera_data in sample_cameras:
            try:
                # Check if camera already exists
                existing_camera = CameraService.get_camera(db, camera_data["id"])
                if not existing_camera:
                    camera_create = CameraCreate(**camera_data)
                    CameraService.create_camera(db, camera_create)
                    logger.info(f"Created camera: {camera_data['name']}")
                else:
                    logger.info(f"Camera already exists: {camera_data['name']}")
            except Exception as e:
                logger.error(f"Failed to create camera {camera_data['name']}: {e}")

def create_directories():
    """Create necessary directories"""
    directories = ["models", "recordings"]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")
        else:
            logger.info(f"Directory already exists: {directory}")

def main():
    """Main initialization function"""
    logger.info("Starting database initialization...")
    
    # Test database connection
    if not test_connection():
        logger.error("Database connection failed! Please check your database configuration.")
        sys.exit(1)
    
    logger.info("Database connection successful!")
    
    try:
        # Initialize database tables
        logger.info("Creating database tables...")
        init_database()
        logger.info("Database tables created successfully!")
        
        # Create directories
        logger.info("Creating necessary directories...")
        create_directories()
        
        # Create default models
        logger.info("Creating default models...")
        create_default_models()
        
        # Create sample cameras
        logger.info("Creating sample cameras...")
        create_sample_cameras()
        
        logger.info("Database initialization completed successfully!")
        
        print("\n" + "="*60)
        print("DATABASE INITIALIZATION COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nNext steps:")
        print("1. Download YOLO model files to the ./models/ directory")
        print("2. Configure your cameras in the database or via the API")
        print("3. Start the application with: python main.py")
        print("4. Access the API documentation at: http://localhost:8001/docs")
        print("\nDefault models created:")
        print("- yolov8n (Nano) - Fast, good for real-time")
        print("- yolov8s (Small) - Balanced speed and accuracy")
        print("- yolov8m (Medium) - Good accuracy")
        print("- yolov8l (Large) - High accuracy")
        print("- yolov8x (Extra Large) - Highest accuracy")
        print("\nSample cameras created:")
        print("- webcam_0 (Default Webcam) - Enabled")
        print("- ip_camera_1 (IP Camera Example) - Disabled (configure URL)")
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()