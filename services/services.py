# services.py - Fixed version
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_, desc, asc
from typing import List, Optional, Dict, Any, Union, Tuple
from datetime import datetime, timedelta
import logging
import os
import json

from models import (
    CameraModel, ModelInfoModel, DetectionSessionModel, DetectionModel,
    AlertModel, RecordingModel, SystemStatsModel, UserActivityModel
)
from schemas import (
    CameraCreate, CameraUpdate, Camera,
    ModelInfoCreate, ModelInfoUpdate, ModelInfo,
    DetectionSessionCreate, DetectionSessionUpdate, DetectionSession,
    DetectionCreate, Detection, DetectionFilter,
    AlertCreate, AlertUpdate, Alert, AlertFilter,
    RecordingInfoCreate, RecordingInfoUpdate, RecordingInfo, RecordingFilter,
    SystemStatsCreate, SystemStats,
    UserActivityCreate, UserActivity,
    PaginationParams, PaginatedResponse,
    DetectionStatistics, AlertStatistics, SystemPerformanceStats
)

logger = logging.getLogger(__name__)

class DatabaseService:
    """Base service class with common database operations"""
    
    @staticmethod
    def apply_pagination(query, pagination: PaginationParams):
        """Apply pagination to a query"""
        total = query.count()
        items = query.offset((pagination.page - 1) * pagination.page_size).limit(pagination.page_size).all()
        
        total_pages = (total + pagination.page_size - 1) // pagination.page_size
        
        return PaginatedResponse(
            items=items,
            total=total,
            page=pagination.page,
            page_size=pagination.page_size,
            total_pages=total_pages,
            has_next=pagination.page < total_pages,
            has_prev=pagination.page > 1
        )

class CameraService(DatabaseService):
    """Service for camera-related database operations"""
    
    @staticmethod
    def create_camera(db: Session, camera: CameraCreate) -> CameraModel:
        """Create a new camera"""
        db_camera = CameraModel(
            id=str(camera.id),
            name=camera.name,
            type=camera.type,
            source=camera.source,
            resolution=camera.resolution,
            fps=camera.fps,
            enabled=camera.enabled,
            recording=camera.recording,
            detection_enabled=camera.detection_enabled,
            alert_threshold=camera.alert_threshold,
            config=getattr(camera, 'config', None)
        )
        db.add(db_camera)
        db.commit()
        db.refresh(db_camera)
        logger.info(f"Created camera: {camera.id}")
        return db_camera
    
    @staticmethod
    def get_camera(db: Session, camera_id: str) -> Optional[CameraModel]:
        """Get a camera by ID"""
        return db.query(CameraModel).filter(CameraModel.id == camera_id).first()
    
    @staticmethod
    def get_cameras(db: Session, enabled_only: bool = False) -> List[CameraModel]:
        """Get all cameras"""
        query = db.query(CameraModel)
        if enabled_only:
            query = query.filter(CameraModel.enabled == True)
        return query.all()
    
    @staticmethod
    def update_camera(db: Session, camera_id: str, camera_update: CameraUpdate) -> Optional[CameraModel]:
        """Update a camera"""
        db_camera = db.query(CameraModel).filter(CameraModel.id == camera_id).first()
        if not db_camera:
            return None
        
        update_data = camera_update.model_dump(exclude_unset=True)
        for key, value in update_data.items():
            setattr(db_camera, key, value)
        
        db_camera.updated_at = datetime.now()
        db.commit()
        db.refresh(db_camera)
        logger.info(f"Updated camera: {camera_id}")
        return db_camera
    
    @staticmethod
    def update_camera_status(db: Session, camera_id: str, status: str) -> bool:
        """Update camera status"""
        db_camera = db.query(CameraModel).filter(CameraModel.id == camera_id).first()
        if not db_camera:
            return False
        
        db_camera.status = status
        db_camera.updated_at = datetime.now()
        db.commit()
        return True
    
    @staticmethod
    def delete_camera(db: Session, camera_id: str) -> bool:
        """Delete a camera"""
        db_camera = db.query(CameraModel).filter(CameraModel.id == camera_id).first()
        if not db_camera:
            return False
        
        db.delete(db_camera)
        db.commit()
        logger.info(f"Deleted camera: {camera_id}")
        return True

class ModelService(DatabaseService):
    """Service for model-related database operations"""
    
    @staticmethod
    def create_model(db: Session, model: ModelInfoCreate) -> ModelInfoModel:
        """Create a new model"""
        db_model = ModelInfoModel(**model.model_dump())
        db.add(db_model)
        db.commit()
        db.refresh(db_model)
        logger.info(f"Created model: {model.id}")
        return db_model
    
    @staticmethod
    def get_model(db: Session, model_id: str) -> Optional[ModelInfoModel]:
        """Get a model by ID"""
        return db.query(ModelInfoModel).filter(ModelInfoModel.id == model_id).first()
    
    @staticmethod
    def get_models(db: Session, enabled_only: bool = False) -> List[ModelInfoModel]:
        """Get all models"""
        query = db.query(ModelInfoModel)
        if enabled_only:
            query = query.filter(ModelInfoModel.enabled == True)
        return query.all()
    
    @staticmethod
    def update_model(db: Session, model_id: str, model_update: ModelInfoUpdate) -> Optional[ModelInfoModel]:
        """Update a model"""
        db_model = db.query(ModelInfoModel).filter(ModelInfoModel.id == model_id).first()
        if not db_model:
            return None
        
        update_data = model_update.model_dump(exclude_unset=True)
        for key, value in update_data.items():
            setattr(db_model, key, value)
        
        db_model.updated_at = datetime.now()
        db.commit()
        db.refresh(db_model)
        logger.info(f"Updated model: {model_id}")
        return db_model
    
    @staticmethod
    def update_model_stats(db: Session, model_id: str, inference_time: float) -> bool:
        """Update model performance statistics"""
        db_model = db.query(ModelInfoModel).filter(ModelInfoModel.id == model_id).first()
        if not db_model:
            return False
        
        # Update average inference time
        if db_model.avg_inference_time:
            db_model.avg_inference_time = (
                (db_model.avg_inference_time * db_model.total_inferences + inference_time) /
                (db_model.total_inferences + 1)
            )
        else:
            db_model.avg_inference_time = inference_time
        
        db_model.total_inferences += 1
        db_model.updated_at = datetime.now()
        db.commit()
        return True

class SessionService(DatabaseService):
    """Service for detection session-related database operations"""
    
    @staticmethod
    def create_session(db: Session, session: DetectionSessionCreate) -> DetectionSessionModel:
        """Create a new detection session"""
        session_id = session.session_id or session.pair_id or f"{session.camera_id}_{session.model_id}_{int(datetime.now().timestamp())}"
        
        db_session = DetectionSessionModel(
            id=session_id,
            camera_id=session.camera_id,
            model_id=session.model_id,
            scenario=getattr(session, 'scenario', 'multi-object'),
            enable_multi_model=getattr(session, 'enable_multi_model', False),
            pair_id=getattr(session, 'pair_id', None)
        )
        db.add(db_session)
        db.commit()
        db.refresh(db_session)
        logger.info(f"Created detection session: {session_id}")
        return db_session
    
    @staticmethod
    def get_session(db: Session, session_id: str) -> Optional[DetectionSessionModel]:
        """Get a session by ID"""
        return db.query(DetectionSessionModel).filter(DetectionSessionModel.id == session_id).first()
    
    @staticmethod
    def get_camera_sessions(db: Session, camera_id: str, active_only: bool = True) -> List[DetectionSessionModel]:
        """Get all sessions for a camera"""
        query = db.query(DetectionSessionModel).filter(DetectionSessionModel.camera_id == camera_id)
        if active_only:
            query = query.filter(DetectionSessionModel.active == True)
        return query.all()
    
    @staticmethod
    def get_model_sessions(db: Session, model_id: str, active_only: bool = True) -> List[DetectionSessionModel]:
        """Get all sessions for a model"""
        query = db.query(DetectionSessionModel).filter(DetectionSessionModel.model_id == model_id)
        if active_only:
            query = query.filter(DetectionSessionModel.active == True)
        return query.all()
    
    @staticmethod
    def update_session(db: Session, session_id: str, session_update: DetectionSessionUpdate) -> Optional[DetectionSessionModel]:
        """Update a session"""
        db_session = db.query(DetectionSessionModel).filter(DetectionSessionModel.id == session_id).first()
        if not db_session:
            return None
        
        update_data = session_update.model_dump(exclude_unset=True)
        for key, value in update_data.items():
            setattr(db_session, key, value)
        
        db_session.last_activity = datetime.now()
        db.commit()
        db.refresh(db_session)
        return db_session
    
    @staticmethod
    def update_session_stats(db: Session, session_id: str, frames_processed: int = 1, processing_time: float = None) -> bool:
        """Update session statistics"""
        db_session = db.query(DetectionSessionModel).filter(DetectionSessionModel.id == session_id).first()
        if not db_session:
            return False
        
        db_session.total_frames_processed += frames_processed
        db_session.last_activity = datetime.now()
        
        if processing_time:
            if db_session.avg_processing_time:
                total_time = db_session.avg_processing_time * (db_session.total_frames_processed - frames_processed)
                db_session.avg_processing_time = (total_time + processing_time) / db_session.total_frames_processed
            else:
                db_session.avg_processing_time = processing_time
        
        db.commit()
        return True
    
    @staticmethod
    def stop_session(db: Session, session_id: str) -> bool:
        """Stop a detection session"""
        db_session = db.query(DetectionSessionModel).filter(DetectionSessionModel.id == session_id).first()
        if not db_session:
            return False
        
        db_session.active = False
        db_session.status = "inactive"
        db_session.stopped_at = datetime.now()
        db.commit()
        logger.info(f"Stopped detection session: {session_id}")
        return True
    
    @staticmethod
    def cleanup_inactive_sessions(db: Session, hours: int = 24) -> int:
        """Clean up inactive sessions older than specified hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        sessions_to_delete = db.query(DetectionSessionModel).filter(
            and_(
                DetectionSessionModel.active == False,
                DetectionSessionModel.last_activity < cutoff_time
            )
        ).all()
        
        count = len(sessions_to_delete)
        for session in sessions_to_delete:
            db.delete(session)
        
        db.commit()
        logger.info(f"Cleaned up {count} inactive sessions")
        return count

class DetectionService(DatabaseService):
    """Service for detection-related database operations"""
    
    @staticmethod
    def create_detection(db: Session, detection: DetectionCreate) -> DetectionModel:
        """Create a new detection"""
        db_detection = DetectionModel(
            session_id=detection.session_id,
            camera_id=detection.camera_id,
            model_id=detection.model_id,
            bbox_x1=detection.bbox[0],
            bbox_y1=detection.bbox[1],
            bbox_x2=detection.bbox[2],
            bbox_y2=detection.bbox[3],
            confidence=detection.confidence,
            class_id=detection.class_id,
            class_name=detection.class_name,
            frame_width=getattr(detection, 'frame_width', None),
            frame_height=getattr(detection, 'frame_height', None),
            frame_timestamp=detection.frame_timestamp,
            inference_time=getattr(detection, 'inference_time', None),
            meta_data=getattr(detection, 'meta_data', None)
        )
        db.add(db_detection)
        db.commit()
        db.refresh(db_detection)
        
        # Update session detection count
        db.query(DetectionSessionModel).filter(DetectionSessionModel.id == detection.session_id).update({
            "total_detections": DetectionSessionModel.total_detections + 1
        })
        db.commit()
        
        return db_detection
    
    @staticmethod
    def get_detections(
        db: Session, 
        filters: DetectionFilter = None, 
        pagination: PaginationParams = None
    ) -> Union[List[DetectionModel], PaginatedResponse]:
        """Get detections with optional filtering and pagination"""
        query = db.query(DetectionModel)
        
        if filters:
            if filters.max_duration:
                query = query.filter(RecordingModel.duration_seconds <= filters.max_duration)
        
        query = query.order_by(desc(RecordingModel.started_at))
        
        if pagination:
            return DatabaseService.apply_pagination(query, pagination)
        
        return query.all()
    
    @staticmethod
    def update_recording(db: Session, recording_id: str, recording_update: RecordingInfoUpdate) -> Optional[RecordingModel]:
        """Update a recording"""
        db_recording = db.query(RecordingModel).filter(RecordingModel.id == recording_id).first()
        if not db_recording:
            return None
        
        update_data = recording_update.model_dump(exclude_unset=True)
        for key, value in update_data.items():
            setattr(db_recording, key, value)
        
        db.commit()
        db.refresh(db_recording)
        return db_recording
    
    @staticmethod
    def delete_recording(db: Session, recording_id: str) -> bool:
        """Delete a recording"""
        db_recording = db.query(RecordingModel).filter(RecordingModel.id == recording_id).first()
        if not db_recording:
            return False
        
        # Delete file if it exists
        if db_recording.file_exists and os.path.exists(db_recording.file_path):
            try:
                os.remove(db_recording.file_path)
                logger.info(f"Deleted recording file: {db_recording.file_path}")
            except Exception as e:
                logger.error(f"Failed to delete recording file: {e}")
        
        db.delete(db_recording)
        db.commit()
        logger.info(f"Deleted recording: {recording_id}")
        return True
    
    @staticmethod
    def check_file_existence(db: Session, recording_id: str) -> bool:
        """Check if recording file exists and update database"""
        db_recording = db.query(RecordingModel).filter(RecordingModel.id == recording_id).first()
        if not db_recording:
            return False
        
        file_exists = os.path.exists(db_recording.file_path)
        
        if db_recording.file_exists != file_exists:
            db_recording.file_exists = file_exists
            db_recording.last_checked = datetime.now()
            db.commit()
        
        return file_exists
    
    @staticmethod
    def cleanup_orphaned_recordings(db: Session) -> int:
        """Clean up database entries for recordings whose files no longer exist"""
        recordings = db.query(RecordingModel).filter(RecordingModel.file_exists == True).all()
        
        orphaned_count = 0
        for recording in recordings:
            if not os.path.exists(recording.file_path):
                recording.file_exists = False
                recording.last_checked = datetime.now()
                orphaned_count += 1
        
        db.commit()
        logger.info(f"Found {orphaned_count} orphaned recordings")
        return orphaned_count

class SystemStatsService(DatabaseService):
    """Service for system statistics"""
    
    @staticmethod
    def create_stats(db: Session, stats: SystemStatsCreate) -> SystemStatsModel:
        """Create new system statistics entry"""
        db_stats = SystemStatsModel(**stats.model_dump())
        db.add(db_stats)
        db.commit()
        db.refresh(db_stats)
        return db_stats
    
    @staticmethod
    def get_latest_stats(db: Session) -> Optional[SystemStatsModel]:
        """Get the latest system statistics"""
        return db.query(SystemStatsModel).order_by(desc(SystemStatsModel.recorded_at)).first()
    
    @staticmethod
    def get_stats_history(
        db: Session, 
        hours: int = 24, 
        pagination: PaginationParams = None
    ) -> Union[List[SystemStatsModel], PaginatedResponse]:
        """Get system statistics history"""
        start_time = datetime.now() - timedelta(hours=hours)
        query = db.query(SystemStatsModel).filter(
            SystemStatsModel.recorded_at >= start_time
        ).order_by(desc(SystemStatsModel.recorded_at))
        
        if pagination:
            return DatabaseService.apply_pagination(query, pagination)
        
        return query.all()
    
    @staticmethod
    def cleanup_old_stats(db: Session, days: int = 90) -> int:
        """Clean up old system statistics"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        stats_to_delete = db.query(SystemStatsModel).filter(
            SystemStatsModel.recorded_at < cutoff_date
        ).all()
        
        count = len(stats_to_delete)
        for stat in stats_to_delete:
            db.delete(stat)
        
        db.commit()
        logger.info(f"Cleaned up {count} old system statistics")
        return count

class UserActivityService(DatabaseService):
    """Service for user activity tracking"""
    
    @staticmethod
    def log_activity(db: Session, activity: UserActivityCreate) -> UserActivityModel:
        """Log a user activity"""
        db_activity = UserActivityModel(**activity.model_dump())
        db.add(db_activity)
        db.commit()
        db.refresh(db_activity)
        return db_activity
    
    @staticmethod
    def get_activities(
        db: Session, 
        action: str = None, 
        resource_type: str = None,
        hours: int = 24,
        pagination: PaginationParams = None
    ) -> Union[List[UserActivityModel], PaginatedResponse]:
        """Get user activities with optional filtering"""
        start_time = datetime.now() - timedelta(hours=hours)
        query = db.query(UserActivityModel).filter(
            UserActivityModel.timestamp >= start_time
        )
        
        if action:
            query = query.filter(UserActivityModel.action == action)
        if resource_type:
            query = query.filter(UserActivityModel.resource_type == resource_type)
        
        query = query.order_by(desc(UserActivityModel.timestamp))
        
        if pagination:
            return DatabaseService.apply_pagination(query, pagination)
        
        return query.all()
    
    @staticmethod
    def get_activity_statistics(db: Session, hours: int = 24) -> Dict[str, Any]:
        """Get activity statistics"""
        start_time = datetime.now() - timedelta(hours=hours)
        
        activities = db.query(UserActivityModel).filter(
            UserActivityModel.timestamp >= start_time
        ).all()
        
        # Count by action
        actions_count = {}
        resource_types_count = {}
        status_codes_count = {}
        hourly_activity = {}
        
        for activity in activities:
            # By action
            actions_count[activity.action] = actions_count.get(activity.action, 0) + 1
            
            # By resource type
            resource_types_count[activity.resource_type] = resource_types_count.get(activity.resource_type, 0) + 1
            
            # By status code
            if activity.status_code:
                status_codes_count[str(activity.status_code)] = status_codes_count.get(str(activity.status_code), 0) + 1
            
            # By hour
            hour_key = activity.timestamp.strftime("%Y-%m-%d %H:00")
            hourly_activity[hour_key] = hourly_activity.get(hour_key, 0) + 1
        
        return {
            "total_activities": len(activities),
            "actions_count": actions_count,
            "resource_types_count": resource_types_count,
            "status_codes_count": status_codes_count,
            "hourly_activity": hourly_activity,
            "period_hours": hours
        }
    
    @staticmethod
    def cleanup_old_activities(db: Session, days: int = 30) -> int:
        """Clean up old user activities"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        activities_to_delete = db.query(UserActivityModel).filter(
            UserActivityModel.timestamp < cutoff_date
        ).all()
        
        count = len(activities_to_delete)
        for activity in activities_to_delete:
            db.delete(activity)
        
        db.commit()
        logger.info(f"Cleaned up {count} old user activities")
        return count

class DatabaseMaintenanceService:
    """Service for database maintenance and cleanup operations"""
    
    @staticmethod
    def run_maintenance(db: Session, config: Dict[str, Any] = None) -> Dict[str, int]:
        """Run comprehensive database maintenance"""
        if not config:
            config = {
                "detection_retention_days": 30,
                "alert_retention_days": 90,
                "session_cleanup_hours": 24,
                "stats_retention_days": 90,
                "activity_retention_days": 30
            }
        
        results = {}
        
        # Clean up old detections
        if config.get("detection_retention_days"):
            results["detections_cleaned"] = DetectionService.cleanup_old_detections(
                db, config["detection_retention_days"]
            )
        
        # Clean up inactive sessions
        if config.get("session_cleanup_hours"):
            results["sessions_cleaned"] = SessionService.cleanup_inactive_sessions(
                db, config["session_cleanup_hours"]
            )
        
        # Clean up old system stats
        if config.get("stats_retention_days"):
            results["stats_cleaned"] = SystemStatsService.cleanup_old_stats(
                db, config["stats_retention_days"]
            )
        
        # Clean up old user activities
        if config.get("activity_retention_days"):
            results["activities_cleaned"] = UserActivityService.cleanup_old_activities(
                db, config["activity_retention_days"]
            )
        
        # Check for orphaned recordings
        results["orphaned_recordings"] = RecordingService.cleanup_orphaned_recordings(db)
        
        logger.info(f"Database maintenance completed: {results}")
        return results
    
    @staticmethod
    def get_database_size_info(db: Session) -> Dict[str, Any]:
        """Get database size information"""
        try:
            # Get table sizes (MySQL specific)
            table_sizes = db.execute("""
                SELECT 
                    table_name,
                    ROUND(((data_length + index_length) / 1024 / 1024), 2) AS size_mb,
                    table_rows
                FROM information_schema.TABLES 
                WHERE table_schema = DATABASE()
                ORDER BY (data_length + index_length) DESC;
            """).fetchall()
            
            total_size = sum(row[1] for row in table_sizes)
            
            return {
                "total_size_mb": total_size,
                "table_sizes": [
                    {"table": row[0], "size_mb": row[1], "rows": row[2]}
                    for row in table_sizes
                ]
            }
        except Exception as e:
            logger.error(f"Failed to get database size info: {e}")
            return {"error": str(e)}
    
    @staticmethod
    def optimize_tables(db: Session) -> Dict[str, Any]:
        """Optimize database tables"""
        try:
            # Get all tables
            tables = db.execute("SHOW TABLES").fetchall()
            optimized_tables = []
            
            for table in tables:
                table_name = table[0]
                try:
                    db.execute(f"OPTIMIZE TABLE {table_name}")
                    optimized_tables.append(table_name)
                except Exception as e:
                    logger.error(f"Failed to optimize table {table_name}: {e}")
            
            db.commit()
            
            return {
                "success": True,
                "optimized_tables": optimized_tables,
                "total_tables": len(optimized_tables)
            }
        except Exception as e:
            logger.error(f"Failed to optimize tables: {e}")
            return {"success": False, "error": str(e)}.camera_id
           
    
    @staticmethod
    def get_detection_statistics(
        db: Session, 
        start_date: datetime = None, 
        end_date: datetime = None,
        camera_id: str = None
    ) -> DetectionStatistics:
        """Get detection statistics for a time period"""
        if not start_date:
            start_date = datetime.now() - timedelta(days=7)
        if not end_date:
            end_date = datetime.now()
        
        query = db.query(DetectionModel).filter(
            and_(
                DetectionModel.detected_at >= start_date,
                DetectionModel.detected_at <= end_date
            )
        )
        
        if camera_id:
            query = query.filter(DetectionModel.camera_id == camera_id)
        
        detections = query.all()
        
        # Calculate statistics
        total_detections = len(detections)
        detections_by_class = {}
        detections_by_hour = {}
        detections_by_camera = {}
        detections_by_model = {}
        total_confidence = 0
        
        for detection in detections:
            # By class
            detections_by_class[detection.class_name] = detections_by_class.get(detection.class_name, 0) + 1
            
            # By hour
            hour_key = detection.detected_at.strftime("%Y-%m-%d %H:00")
            detections_by_hour[hour_key] = detections_by_hour.get(hour_key, 0) + 1
            
            # By camera
            detections_by_camera[detection.camera_id] = detections_by_camera.get(detection.camera_id, 0) + 1
            
            # By model
            detections_by_model[detection.model_id] = detections_by_model.get(detection.model_id, 0) + 1
            
            total_confidence += detection.confidence
        
        avg_confidence = total_confidence / total_detections if total_detections > 0 else 0
        
        return DetectionStatistics(
            total_detections=total_detections,
            detections_by_class=detections_by_class,
            detections_by_hour=detections_by_hour,
            detections_by_camera=detections_by_camera,
            detections_by_model=detections_by_model,
            avg_confidence=avg_confidence,
            period_start=start_date,
            period_end=end_date
        )
    
    @staticmethod
    def cleanup_old_detections(db: Session, days: int = 30) -> int:
        """Clean up detections older than specified days"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        detections_to_delete = db.query(DetectionModel).filter(
            DetectionModel.detected_at < cutoff_date
        ).all()
        
        count = len(detections_to_delete)
        for detection in detections_to_delete:
            db.delete(detection)
        
        db.commit()
        logger.info(f"Cleaned up {count} old detections")
        return count

class AlertService(DatabaseService):
    """Service for alert-related database operations"""
    
    @staticmethod
    def create_alert(db: Session, alert: AlertCreate) -> AlertModel:
        """Create a new alert"""
        alert_id = f"{alert.camera_id}_{alert.model_id}_{int(datetime.now().timestamp() * 1000)}"
        
        db_alert = AlertModel(
            id=alert_id,
            detection_id=alert.detection_id,
            session_id=alert.session_id,
            camera_id=alert.camera_id,
            model_id=alert.model_id,
            object_type=alert.object_type,
            confidence=alert.confidence,
            bbox_x1=alert.bbox[0],
            bbox_y1=alert.bbox[1],
            bbox_x2=alert.bbox[2],
            bbox_y2=alert.bbox[3],
            image_data=getattr(alert, 'image_data', None),
            image_size=getattr(alert, 'image_size', None)
        )
        db.add(db_alert)
        db.commit()
        db.refresh(db_alert)
        logger.info(f"Created alert: {alert_id}")
        return db_alert
    
    @staticmethod
    def get_alerts(
        db: Session, 
        filters: AlertFilter = None, 
        pagination: PaginationParams = None
    ) -> Union[List[AlertModel], PaginatedResponse]:
        """Get alerts with optional filtering and pagination"""
        query = db.query(AlertModel)
        
        if filters:
            if filters.camera_id:
                query = query.filter(AlertModel.camera_id == filters.camera_id)
            if filters.model_id:
                query = query.filter(AlertModel.model_id == filters.model_id)
            if filters.session_id:
                query = query.filter(AlertModel.session_id == filters.session_id)
            if filters.object_type:
                query = query.filter(AlertModel.object_type == filters.object_type)
            if filters.acknowledged is not None:
                query = query.filter(AlertModel.acknowledged == filters.acknowledged)
            if filters.min_confidence:
                query = query.filter(AlertModel.confidence >= filters.min_confidence)
            if filters.start_date:
                query = query.filter(AlertModel.triggered_at >= filters.start_date)
            if filters.end_date:
                query = query.filter(AlertModel.triggered_at <= filters.end_date)
        
        query = query.order_by(desc(AlertModel.triggered_at))
        
        if pagination:
            return DatabaseService.apply_pagination(query, pagination)
        
        return query.all()
    
    @staticmethod
    def acknowledge_alert(db: Session, alert_id: str, acknowledged_by: str = None) -> bool:
        """Acknowledge an alert"""
        db_alert = db.query(AlertModel).filter(AlertModel.id == alert_id).first()
        if not db_alert:
            return False
        
        db_alert.acknowledged = True
        db_alert.acknowledged_at = datetime.now()
        db_alert.acknowledged_by = acknowledged_by
        db.commit()
        logger.info(f"Acknowledged alert: {alert_id}")
        return True
    
    @staticmethod
    def get_alert_statistics(
        db: Session, 
        start_date: datetime = None, 
        end_date: datetime = None,
        camera_id: str = None
    ) -> AlertStatistics:
        """Get alert statistics for a time period"""
        if not start_date:
            start_date = datetime.now() - timedelta(days=7)
        if not end_date:
            end_date = datetime.now()
        
        query = db.query(AlertModel).filter(
            and_(
                AlertModel.triggered_at >= start_date,
                AlertModel.triggered_at <= end_date
            )
        )
        
        if camera_id:
            query = query.filter(AlertModel.camera_id == camera_id)
        
        alerts = query.all()
        
        # Calculate statistics
        total_alerts = len(alerts)
        alerts_by_type = {}
        alerts_by_hour = {}
        alerts_by_camera = {}
        acknowledged_alerts = 0
        total_response_time = 0
        response_time_count = 0
        
        for alert in alerts:
            # By type
            alerts_by_type[alert.object_type] = alerts_by_type.get(alert.object_type, 0) + 1
            
            # By hour
            hour_key = alert.triggered_at.strftime("%Y-%m-%d %H:00")
            alerts_by_hour[hour_key] = alerts_by_hour.get(hour_key, 0) + 1
            
            # By camera
            alerts_by_camera[alert.camera_id] = alerts_by_camera.get(alert.camera_id, 0) + 1
            
            # Acknowledged count
            if alert.acknowledged:
                acknowledged_alerts += 1
                if alert.acknowledged_at:
                    response_time = (alert.acknowledged_at - alert.triggered_at).total_seconds()
                    total_response_time += response_time
                    response_time_count += 1
        
        avg_response_time = total_response_time / response_time_count if response_time_count > 0 else None
        
        return AlertStatistics(
            total_alerts=total_alerts,
            alerts_by_type=alerts_by_type,
            alerts_by_hour=alerts_by_hour,
            alerts_by_camera=alerts_by_camera,
            acknowledged_alerts=acknowledged_alerts,
            unacknowledged_alerts=total_alerts - acknowledged_alerts,
            avg_response_time=avg_response_time,
            period_start=start_date,
            period_end=end_date
        )

class RecordingService(DatabaseService):
    """Service for recording-related database operations"""
    
    @staticmethod
    def create_recording(db: Session, recording: RecordingInfoCreate) -> RecordingModel:
        """Create a new recording"""
        recording_id = f"{recording.camera_id}_{int(datetime.now().timestamp())}"
        
        db_recording = RecordingModel(
            id=recording_id,
            camera_id=recording.camera_id,
            filename=recording.filename,
            file_path=recording.file_path,
            format=recording.format,
            quality=recording.quality,
            resolution=getattr(recording, 'resolution', None),
            fps=getattr(recording, 'fps', None),
            output_directory=getattr(recording, 'output_directory', None),
            max_file_size_mb=getattr(recording, 'max_file_size_mb', None),
            max_duration_minutes=getattr(recording, 'max_duration_minutes', None),
            meta_data=getattr(recording, 'meta_data', None)
        )
        db.add(db_recording)
        db.commit()
        db.refresh(db_recording)
        logger.info(f"Created recording: {recording_id}")
        return db_recording
    
    @staticmethod
    def get_recording(db: Session, recording_id: str) -> Optional[RecordingModel]:
        """Get a recording by ID"""
        return db.query(RecordingModel).filter(RecordingModel.id == recording_id).first()
    
    @staticmethod
    def get_recordings(
        db: Session, 
        filters: RecordingFilter = None, 
        pagination: PaginationParams = None
    ) -> Union[List[RecordingModel], PaginatedResponse]:
        """Get recordings with optional filtering and pagination"""
        query = db.query(RecordingModel)
        
        if filters:
            if filters.camera_id:
                query = query.filter(RecordingModel.camera_id == filters.camera_id)
            if filters.status:
                query = query.filter(RecordingModel.status == filters.status)
            if filters.start_date:
                query = query.filter(RecordingModel.started_at >= filters.start_date)
            if filters.end_date:
                query = query.filter(RecordingModel.started_at <= filters.end_date)
            if filters.min_duration:
                query = query.filter(RecordingModel.duration_seconds >= filters.min_duration)
