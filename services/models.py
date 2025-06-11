# models.py
from sqlalchemy import Column, Integer, String, Text, Boolean, Float, DateTime, ForeignKey, JSON, BigInteger, Enum
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from database import Base
import enum
from datetime import datetime
from typing import Optional

class CameraType(str, enum.Enum):
    WEBCAM = "webcam"
    IP = "ip"

class CameraStatus(str, enum.Enum):
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"

class RecordingStatus(str, enum.Enum):
    RECORDING = "recording"
    COMPLETED = "completed"
    ERROR = "error"
    STOPPED = "stopped"

class SessionStatus(str, enum.Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"

class CameraModel(Base):
    """Database model for cameras"""
    __tablename__ = "cameras"
    
    id = Column(String(50), primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    type = Column(Enum(CameraType), nullable=False)
    source = Column(String(500), nullable=False)  # Webcam index or IP URL
    resolution = Column(String(20), nullable=True)  # e.g., "1920x1080"
    fps = Column(Integer, default=30)
    enabled = Column(Boolean, default=True)
    recording = Column(Boolean, default=False)
    detection_enabled = Column(Boolean, default=True)
    alert_threshold = Column(Float, default=0.5)
    status = Column(Enum(CameraStatus), default=CameraStatus.DISCONNECTED)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Configuration JSON for additional settings
    config = Column(JSON, nullable=True)
    
    # Relationships
    detection_sessions = relationship("DetectionSessionModel", back_populates="camera", cascade="all, delete-orphan")
    recordings = relationship("RecordingModel", back_populates="camera", cascade="all, delete-orphan")
    detections = relationship("DetectionModel", back_populates="camera", cascade="all, delete-orphan")
    alerts = relationship("AlertModel", back_populates="camera", cascade="all, delete-orphan")

class ModelInfoModel(Base):
    """Database model for YOLO models"""
    __tablename__ = "models"
    
    id = Column(String(50), primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    path = Column(String(500), nullable=False)
    version = Column(String(50), nullable=True)
    description = Column(Text, nullable=True)
    enabled = Column(Boolean, default=True)
    
    # Model metadata
    input_size = Column(String(20), nullable=True)  # e.g., "640x640"
    classes = Column(JSON, nullable=True)  # List of class names
    confidence_threshold = Column(Float, default=0.25)
    
    # Performance metrics
    avg_inference_time = Column(Float, nullable=True)
    total_inferences = Column(BigInteger, default=0)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    detection_sessions = relationship("DetectionSessionModel", back_populates="model", cascade="all, delete-orphan")
    detections = relationship("DetectionModel", back_populates="model", cascade="all, delete-orphan")
    alerts = relationship("AlertModel", back_populates="model", cascade="all, delete-orphan")

class DetectionSessionModel(Base):
    """Database model for detection sessions"""
    __tablename__ = "detection_sessions"
    
    id = Column(String(100), primary_key=True, index=True)
    camera_id = Column(String(50), ForeignKey("cameras.id"), nullable=False)
    model_id = Column(String(50), ForeignKey("models.id"), nullable=False)
    
    # Session configuration
    scenario = Column(String(100), default="multi-object")
    enable_multi_model = Column(Boolean, default=False)
    pair_id = Column(String(100), nullable=True, index=True)  # For pairing with external systems
    
    # Session status
    status = Column(Enum(SessionStatus), default=SessionStatus.ACTIVE)
    active = Column(Boolean, default=True)
    
    # Statistics
    total_detections = Column(BigInteger, default=0)
    total_frames_processed = Column(BigInteger, default=0)
    avg_processing_time = Column(Float, nullable=True)
    websocket_clients_count = Column(Integer, default=0)
    
    # Timestamps
    started_at = Column(DateTime(timezone=True), server_default=func.now())
    stopped_at = Column(DateTime(timezone=True), nullable=True)
    last_activity = Column(DateTime(timezone=True), server_default=func.now())
    
    # Error tracking
    error_count = Column(Integer, default=0)
    last_error = Column(Text, nullable=True)
    last_error_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    camera = relationship("CameraModel", back_populates="detection_sessions")
    model = relationship("ModelInfoModel", back_populates="detection_sessions")
    detections = relationship("DetectionModel", back_populates="session", cascade="all, delete-orphan")
    alerts = relationship("AlertModel", back_populates="session", cascade="all, delete-orphan")

class DetectionModel(Base):
    """Database model for object detections"""
    __tablename__ = "detections"
    
    id = Column(BigInteger, primary_key=True, index=True, autoincrement=True)
    session_id = Column(String(100), ForeignKey("detection_sessions.id"), nullable=False)
    camera_id = Column(String(50), ForeignKey("cameras.id"), nullable=False)
    model_id = Column(String(50), ForeignKey("models.id"), nullable=False)
    
    # Detection data
    bbox_x1 = Column(Float, nullable=False)
    bbox_y1 = Column(Float, nullable=False)
    bbox_x2 = Column(Float, nullable=False)
    bbox_y2 = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)
    class_id = Column(Integer, nullable=False)
    class_name = Column(String(100), nullable=False)
    
    # Frame information
    frame_width = Column(Integer, nullable=True)
    frame_height = Column(Integer, nullable=True)
    frame_timestamp = Column(DateTime(timezone=True), nullable=False)
    
    # Processing metadata
    inference_time = Column(Float, nullable=True)  # Time taken for inference in seconds
    detected_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    
    # Additional data
    meta_data = Column(JSON, nullable=True)  # For storing additional detection metadata
    
    # Relationships
    session = relationship("DetectionSessionModel", back_populates="detections")
    camera = relationship("CameraModel", back_populates="detections")
    model = relationship("ModelInfoModel", back_populates="detections")
    alerts = relationship("AlertModel", back_populates="detection", cascade="all, delete-orphan")

class AlertModel(Base):
    """Database model for detection alerts"""
    __tablename__ = "alerts"
    
    id = Column(String(100), primary_key=True, index=True)
    detection_id = Column(BigInteger, ForeignKey("detections.id"), nullable=False)
    session_id = Column(String(100), ForeignKey("detection_sessions.id"), nullable=False)
    camera_id = Column(String(50), ForeignKey("cameras.id"), nullable=False)
    model_id = Column(String(50), ForeignKey("models.id"), nullable=False)
    
    # Alert data
    object_type = Column(String(100), nullable=False)
    confidence = Column(Float, nullable=False)
    bbox_x1 = Column(Float, nullable=False)
    bbox_y1 = Column(Float, nullable=False)
    bbox_x2 = Column(Float, nullable=False)
    bbox_y2 = Column(Float, nullable=False)
    
    # Alert status
    acknowledged = Column(Boolean, default=False)
    acknowledged_at = Column(DateTime(timezone=True), nullable=True)
    acknowledged_by = Column(String(100), nullable=True)
    
    # Image data (base64 encoded)
    image_data = Column(Text, nullable=True)  # Store as TEXT for large base64 strings
    image_size = Column(Integer, nullable=True)  # Size in bytes
    
    # Timestamps
    triggered_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    
    # Notification tracking
    notification_sent = Column(Boolean, default=False)
    notification_sent_at = Column(DateTime(timezone=True), nullable=True)
    notification_channels = Column(JSON, nullable=True)  # Track which channels were notified
    
    # Relationships
    detection = relationship("DetectionModel", back_populates="alerts")
    session = relationship("DetectionSessionModel", back_populates="alerts")
    camera = relationship("CameraModel", back_populates="alerts")
    model = relationship("ModelInfoModel", back_populates="alerts")

class RecordingModel(Base):
    """Database model for camera recordings"""
    __tablename__ = "recordings"
    
    id = Column(String(100), primary_key=True, index=True)
    camera_id = Column(String(50), ForeignKey("cameras.id"), nullable=False)
    
    # Recording metadata
    filename = Column(String(500), nullable=False)
    file_path = Column(String(1000), nullable=False)
    format = Column(String(10), default="mp4")
    quality = Column(Integer, default=80)
    
    # Recording details
    resolution = Column(String(20), nullable=True)
    fps = Column(Float, nullable=True)
    duration_seconds = Column(Float, nullable=True)
    file_size_bytes = Column(BigInteger, nullable=True)
    
    # Status and timestamps
    status = Column(Enum(RecordingStatus), default=RecordingStatus.RECORDING)
    started_at = Column(DateTime(timezone=True), server_default=func.now())
    stopped_at = Column(DateTime(timezone=True), nullable=True)
    
    # File system tracking
    file_exists = Column(Boolean, default=True)
    last_checked = Column(DateTime(timezone=True), nullable=True)
    
    # Configuration used
    output_directory = Column(String(1000), nullable=True)
    max_file_size_mb = Column(Integer, nullable=True)
    max_duration_minutes = Column(Integer, nullable=True)
    
    # Error tracking
    error_message = Column(Text, nullable=True)
    error_count = Column(Integer, default=0)
    
    # Metadata
    meta_data = Column(JSON, nullable=True)
    
    # Relationships
    camera = relationship("CameraModel", back_populates="recordings")

class SystemStatsModel(Base):
    """Database model for system statistics and monitoring"""
    __tablename__ = "system_stats"
    
    id = Column(BigInteger, primary_key=True, index=True, autoincrement=True)
    
    # System metrics
    active_cameras = Column(Integer, default=0)
    total_cameras = Column(Integer, default=0)
    active_sessions = Column(Integer, default=0)
    total_detections_today = Column(BigInteger, default=0)
    total_alerts_today = Column(BigInteger, default=0)
    active_recordings = Column(Integer, default=0)
    
    # Performance metrics
    avg_processing_time = Column(Float, nullable=True)
    total_frames_processed = Column(BigInteger, default=0)
    memory_usage_mb = Column(Float, nullable=True)
    cpu_usage_percent = Column(Float, nullable=True)
    
    # Storage metrics
    total_storage_used_mb = Column(BigInteger, default=0)
    recordings_storage_mb = Column(BigInteger, default=0)
    models_storage_mb = Column(BigInteger, default=0)
    
    # Network metrics
    websocket_connections = Column(Integer, default=0)
    api_requests_today = Column(BigInteger, default=0)
    
    # Timestamp
    recorded_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    
    # Additional metadata
    version = Column(String(50), nullable=True)
    uptime_seconds = Column(BigInteger, nullable=True)
    meta_data = Column(JSON, nullable=True)

class UserActivityModel(Base):
    """Database model for tracking user activities and API usage"""
    __tablename__ = "user_activities"
    
    id = Column(BigInteger, primary_key=True, index=True, autoincrement=True)
    
    # Activity details
    action = Column(String(100), nullable=False, index=True)  # e.g., "start_tracking", "stop_recording"
    resource_type = Column(String(50), nullable=False)  # e.g., "camera", "session", "recording"
    resource_id = Column(String(100), nullable=True)  # ID of the resource being acted upon
    
    # Request details
    endpoint = Column(String(500), nullable=True)
    method = Column(String(10), nullable=True)  # GET, POST, etc.
    status_code = Column(Integer, nullable=True)
    
    # User context
    user_agent = Column(String(1000), nullable=True)
    ip_address = Column(String(45), nullable=True)  # IPv6 support
    session_id = Column(String(100), nullable=True)
    
    # Timing
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    response_time_ms = Column(Float, nullable=True)
    
    # Additional data
    request_data = Column(JSON, nullable=True)
    response_data = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)

# Indexes for better performance
from sqlalchemy import Index

# Create composite indexes for better query performance
Index('idx_detections_camera_time', DetectionModel.camera_id, DetectionModel.detected_at)
Index('idx_detections_session_time', DetectionModel.session_id, DetectionModel.detected_at)
Index('idx_alerts_camera_time', AlertModel.camera_id, AlertModel.triggered_at)
Index('idx_alerts_acknowledged', AlertModel.acknowledged, AlertModel.triggered_at)
Index('idx_recordings_camera_status', RecordingModel.camera_id, RecordingModel.status)
Index('idx_sessions_camera_status', DetectionSessionModel.camera_id, DetectionSessionModel.status)
Index('idx_user_activities_action_time', UserActivityModel.action, UserActivityModel.timestamp)