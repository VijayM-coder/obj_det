# schemas.py
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Any, Union
from datetime import datetime
from enum import Enum

# Enums
class CameraType(str, Enum):
    WEBCAM = "webcam"
    IP = "ip"

class CameraStatus(str, Enum):
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"

class RecordingStatus(str, Enum):
    RECORDING = "recording"
    COMPLETED = "completed"
    ERROR = "error"
    STOPPED = "stopped"

class SessionStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"

class CameraBase(BaseModel):
    id: Union[int, str]
    name: str
    type: CameraType
    source: str
    resolution: Optional[str] = None
    fps: int = 30
    enabled: bool = True
    recording: bool = False
    detection_enabled: bool = True
    alert_threshold: float = 0.5

    model_config = {"protected_namespaces": ()}

class CameraCreate(CameraBase):
    config: Optional[Dict[str, Any]] = None

    model_config = {"protected_namespaces": ()}

class CameraUpdate(BaseModel):
    name: Optional[str] = None
    type: Optional[CameraType] = None
    source: Optional[str] = None
    resolution: Optional[str] = None
    fps: Optional[int] = None
    enabled: Optional[bool] = None
    recording: Optional[bool] = None
    detection_enabled: Optional[bool] = None
    alert_threshold: Optional[float] = None
    config: Optional[Dict[str, Any]] = None

    model_config = {"protected_namespaces": ()}

class Camera(CameraBase):
    status: CameraStatus = CameraStatus.DISCONNECTED
    created_at: datetime
    updated_at: Optional[datetime] = None
    config: Optional[Dict[str, Any]] = None

    model_config = {"from_attributes": True, "protected_namespaces": ()}

class ModelInfoBase(BaseModel):
    id: str
    name: str
    path: str
    version: Optional[str] = None
    description: Optional[str] = None
    enabled: bool = True
    input_size: Optional[str] = None
    classes: Optional[List[str]] = None
    confidence_threshold: float = 0.25

    model_config = {"protected_namespaces": ()}

class ModelInfoCreate(ModelInfoBase):
    model_config = {"protected_namespaces": ()}

class ModelInfoUpdate(BaseModel):
    name: Optional[str] = None
    path: Optional[str] = None
    version: Optional[str] = None
    description: Optional[str] = None
    enabled: Optional[bool] = None
    input_size: Optional[str] = None
    classes: Optional[List[str]] = None
    confidence_threshold: Optional[float] = None

    model_config = {"protected_namespaces": ()}

class ModelInfo(ModelInfoBase):
    avg_inference_time: Optional[float] = None
    total_inferences: int = 0
    created_at: datetime
    updated_at: Optional[datetime] = None

    model_config = {"from_attributes": True, "protected_namespaces": ()}

# Detection Session schemas
class DetectionSessionBase(BaseModel):
    id: str
    camera_id: str
    model_id: str
    scenario: str = "multi-object"
    enable_multi_model: bool = False
    pair_id: Optional[str] = None

    model_config = {"protected_namespaces": ()}

class DetectionSessionCreate(BaseModel):
    camera_id: str
    model_id: str
    session_id: Optional[str] = None
    scenario: Optional[str] = "multi-object"
    enable_multi_model: Optional[bool] = False
    pair_id: Optional[str] = None

    model_config = {"protected_namespaces": ()}

class DetectionSessionUpdate(BaseModel):
    scenario: Optional[str] = None
    enable_multi_model: Optional[bool] = None
    active: Optional[bool] = None
    status: Optional[SessionStatus] = None
    error_count: Optional[int] = None
    last_error: Optional[str] = None
    last_error_at: Optional[datetime] = None

    model_config = {"protected_namespaces": ()}

class DetectionSession(DetectionSessionBase):
    status: SessionStatus = SessionStatus.ACTIVE
    active: bool = True
    total_detections: int = 0
    total_frames_processed: int = 0
    avg_processing_time: Optional[float] = None
    websocket_clients_count: int = 0
    started_at: datetime
    stopped_at: Optional[datetime] = None
    last_activity: datetime
    error_count: int = 0
    last_error: Optional[str] = None
    last_error_at: Optional[datetime] = None

    model_config = {"from_attributes": True, "protected_namespaces": ()}

class DetectionBase(BaseModel):
    bbox: List[float] = Field(..., min_items=4, max_items=4)
    confidence: float = Field(..., ge=0.0, le=1.0)
    class_id: int
    class_name: str
    frame_timestamp: datetime

    @validator('bbox')
    def validate_bbox(cls, v):
        if len(v) != 4:
            raise ValueError('bbox must contain exactly 4 values [x1, y1, x2, y2]')
        return v

    model_config = {"protected_namespaces": ()}

class DetectionCreate(DetectionBase):
    session_id: str
    camera_id: str
    model_id: str
    frame_width: Optional[int] = None
    frame_height: Optional[int] = None
    inference_time: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

class Detection(DetectionBase):
    id: int
    session_id: str
    camera_id: str
    model_id: str
    bbox_x1: float
    bbox_y1: float
    bbox_x2: float
    bbox_y2: float
    frame_width: Optional[int] = None
    frame_height: Optional[int] = None
    inference_time: Optional[float] = None
    detected_at: datetime
    meta_data: Optional[Dict[str, Any]] = None

    model_config = {"from_attributes": True, "protected_namespaces": ()}

class AlertBase(BaseModel):
    object_type: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox: List[float] = Field(..., min_items=4, max_items=4)

    model_config = {"protected_namespaces": ()}

class AlertCreate(AlertBase):
    detection_id: int
    session_id: str
    camera_id: str
    model_id: str
    image_data: Optional[str] = None
    image_size: Optional[int] = None

    model_config = {"protected_namespaces": ()}

class AlertUpdate(BaseModel):
    acknowledged: Optional[bool] = None
    acknowledged_by: Optional[str] = None
    notification_sent: Optional[bool] = None
    notification_channels: Optional[List[str]] = None

    model_config = {"protected_namespaces": ()}

class Alert(AlertBase):
    id: str
    detection_id: int
    session_id: str
    camera_id: str
    model_id: str
    bbox_x1: float
    bbox_y1: float
    bbox_x2: float
    bbox_y2: float
    acknowledged: bool = False
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    image_data: Optional[str] = None
    image_size: Optional[int] = None
    triggered_at: datetime
    notification_sent: bool = False
    notification_sent_at: Optional[datetime] = None
    notification_channels: Optional[List[str]] = None

    model_config = {"from_attributes": True, "protected_namespaces": ()}

# Recording schemas
class RecordingConfigBase(BaseModel):
    output_directory: Optional[str] = None
    format: str = "mp4"
    quality: int = 80
    max_file_size_mb: Optional[int] = None
    max_duration_minutes: Optional[int] = None

class RecordingConfig(RecordingConfigBase):
    model_config = {"from_attributes": True, "protected_namespaces": ()}

class RecordingInfoBase(BaseModel):
    camera_id: str
    filename: str
    file_path: str
    format: str = "mp4"
    quality: int = 80

class RecordingInfoCreate(RecordingInfoBase):
    resolution: Optional[str] = None
    fps: Optional[float] = None
    output_directory: Optional[str] = None
    max_file_size_mb: Optional[int] = None
    max_duration_minutes: Optional[int] = None
    meta_data: Optional[Dict[str, Any]] = None

class RecordingInfoUpdate(BaseModel):
    status: Optional[RecordingStatus] = None
    duration_seconds: Optional[float] = None
    file_size_bytes: Optional[int] = None
    stopped_at: Optional[datetime] = None
    file_exists: Optional[bool] = None
    error_message: Optional[str] = None

class RecordingInfo(RecordingInfoBase):
    id: str
    resolution: Optional[str] = None
    fps: Optional[float] = None
    duration_seconds: Optional[float] = None
    file_size_bytes: Optional[int] = None
    status: RecordingStatus = RecordingStatus.RECORDING
    started_at: datetime
    stopped_at: Optional[datetime] = None
    file_exists: bool = True
    last_checked: Optional[datetime] = None
    output_directory: Optional[str] = None
    max_file_size_mb: Optional[int] = None
    max_duration_minutes: Optional[int] = None
    error_message: Optional[str] = None
    error_count: int = 0
    meta_data: Optional[Dict[str, Any]] = None

    model_config = {"from_attributes": True, "protected_namespaces": ()}

# Request/Response schemas
class TrackingRequest(BaseModel):
    model: str
    session_id: Optional[str] = None
    pair_id: Optional[str] = None
    scenario: Optional[str] = "multi-object"
    enable_multi_model: Optional[bool] = False

    model_config = {"protected_namespaces": ()}

class TrackingResponse(BaseModel):
    session_id: str
    camera_id: str
    model: str
    message: str
    success: bool = True

    model_config = {"protected_namespaces": ()}

class DetectionStatusResponse(BaseModel):
    camera_id: str
    detection_enabled: bool
    camera_status: CameraStatus
    active_sessions: int
    active_models: List[str]
    total_detections: int
    recent_detections: int
    last_detection: Optional[Detection] = None
    session_details: List[Dict[str, Any]]

class MultiModelStatusResponse(BaseModel):
    camera_id: str
    total_sessions: int
    models: List[str]
    active_models: List[str]
    sessions: List[Dict[str, Any]]

class SystemStatsBase(BaseModel):
    active_cameras: int = 0
    total_cameras: int = 0
    active_sessions: int = 0
    total_detections_today: int = 0
    total_alerts_today: int = 0
    active_recordings: int = 0
    avg_processing_time: Optional[float] = None
    total_frames_processed: int = 0
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None
    total_storage_used_mb: int = 0
    recordings_storage_mb: int = 0
    models_storage_mb: int = 0
    websocket_connections: int = 0
    api_requests_today: int = 0

class SystemStatsCreate(SystemStatsBase):
    version: Optional[str] = None
    uptime_seconds: Optional[int] = None
    meta_data: Optional[Dict[str, Any]] = None

class SystemStats(SystemStatsBase):
    id: int
    recorded_at: datetime
    version: Optional[str] = None
    uptime_seconds: Optional[int] = None
    meta_data: Optional[Dict[str, Any]] = None

    model_config = {"from_attributes": True, "protected_namespaces": ()}

class UserActivityBase(BaseModel):
    action: str
    resource_type: str
    resource_id: Optional[str] = None
    endpoint: Optional[str] = None
    method: Optional[str] = None
    status_code: Optional[int] = None
    user_agent: Optional[str] = None
    ip_address: Optional[str] = None
    session_id: Optional[str] = None
    response_time_ms: Optional[float] = None
    request_data: Optional[Dict[str, Any]] = None
    response_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

class UserActivityCreate(UserActivityBase):
    pass

class UserActivity(UserActivityBase):
    id: int
    timestamp: datetime

    class Config:
        from_attributes = True

# Health check schemas
class HealthResponse(BaseModel):
    status: str = "ok"
    active_cameras: int
    total_cameras: int
    multi_model_support: bool = True
    active_sessions: int
    loaded_models: int
    recording_support: bool = True
    cameras_recording: int
    total_recordings: int
    database_connected: bool = True
    timestamp: datetime = Field(default_factory=datetime.now)

class DetectionHealthResponse(BaseModel):
    multi_model_support: bool = True
    loaded_models: List[str]
    total_cameras: int
    active_cameras: int
    detection_enabled_cameras: int
    active_sessions: int
    total_websocket_connections: int
    system_status: str = "healthy"
    database_connected: bool = True
    timestamp: datetime = Field(default_factory=datetime.now)

# Pagination schemas
class PaginationParams(BaseModel):
    page: int = Field(1, ge=1)
    page_size: int = Field(50, ge=1, le=1000)

class PaginatedResponse(BaseModel):
    items: List[Any]
    total: int
    page: int
    page_size: int
    total_pages: int
    has_next: bool
    has_prev: bool

# Filter schemas
class DetectionFilter(BaseModel):
    camera_id: Optional[str] = None
    model_id: Optional[str] = None
    session_id: Optional[str] = None
    class_name: Optional[str] = None
    min_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None

    model_config = {"protected_namespaces": ()}

class AlertFilter(BaseModel):
    camera_id: Optional[str] = None
    model_id: Optional[str] = None
    session_id: Optional[str] = None
    object_type: Optional[str] = None
    acknowledged: Optional[bool] = None
    min_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None

    model_config = {"protected_namespaces": ()}

class RecordingFilter(BaseModel):
    camera_id: Optional[str] = None
    status: Optional[RecordingStatus] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    min_duration: Optional[float] = None
    max_duration: Optional[float] = None

# Bulk operation schemas
class BulkOperationResponse(BaseModel):
    success_count: int
    error_count: int
    total_processed: int
    errors: List[Dict[str, Any]] = []
    successful_items: List[str] = []

class BulkDeleteRequest(BaseModel):
    ids: List[Union[str, int]]
    confirm: bool = False

# Statistics and analytics schemas
class DetectionStatistics(BaseModel):
    total_detections: int
    detections_by_class: Dict[str, int]
    detections_by_hour: Dict[str, int]
    detections_by_camera: Dict[str, int]
    detections_by_model: Dict[str, int]
    avg_confidence: float
    period_start: datetime
    period_end: datetime

class AlertStatistics(BaseModel):
    total_alerts: int
    alerts_by_type: Dict[str, int]
    alerts_by_hour: Dict[str, int]
    alerts_by_camera: Dict[str, int]
    acknowledged_alerts: int
    unacknowledged_alerts: int
    avg_response_time: Optional[float] = None
    period_start: datetime
    period_end: datetime

class SystemPerformanceStats(BaseModel):
    avg_detection_time: float
    avg_frame_processing_time: float
    frames_per_second: float
    detection_accuracy: Optional[float] = None
    system_uptime: int
    memory_usage: float
    cpu_usage: float
    storage_usage: float
    network_usage: Optional[float] = None

# WebSocket message schemas
class WebSocketMessage(BaseModel):
    type: str
    timestamp: datetime = Field(default_factory=datetime.now)
    data: Optional[Dict[str, Any]] = None

class AlertWebSocketMessage(WebSocketMessage):
    type: str = "alert"
    camera_id: str
    model: str
    session_id: str
    alert: Alert

class StatusWebSocketMessage(WebSocketMessage):
    type: str = "status"
    camera_id: str
    status: Dict[str, Any]

class ConnectionWebSocketMessage(WebSocketMessage):
    type: str = "connection"
    status: str
    camera_id: str
    model: Optional[str] = None
    message: str

# Export schemas for data export functionality
class ExportRequest(BaseModel):
    export_type: str  # "detections", "alerts", "recordings", "stats"
    format: str = "json"  # "json", "csv", "excel"
    filters: Optional[Dict[str, Any]] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    include_images: bool = False

class ExportResponse(BaseModel):
    export_id: str
    status: str
    file_path: Optional[str] = None
    file_size_bytes: Optional[int] = None
    created_at: datetime
    expires_at: Optional[datetime] = None
    download_url: Optional[str] = None