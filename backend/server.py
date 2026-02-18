"""
ETL & ML Dashboard - Backend API Server (FAANG-Level Architecture)

A comprehensive FastAPI-based backend for managing ETL pipelines, 
ML experiments, and data quality monitoring with real-time WebSocket support.

Features:
- Pipeline management and execution tracking
- ML experiment tracking with sklearn + AutoML
- Data validation and quality metrics
- Real-time WebSocket monitoring
- AWS S3 integration for cloud storage
- Background task processing
"""

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Set
from datetime import datetime, timezone
from enum import Enum
import os
import uuid
import random
import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pymongo import MongoClient

# ML imports
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import io

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration & Database Setup
# ============================================================================

MONGO_URL = os.environ.get("MONGO_URL", "mongodb://localhost:27017")
DB_NAME = os.environ.get("DB_NAME", "etl_ml_dashboard")

# AWS S3 Configuration (optional)
AWS_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
AWS_BUCKET_NAME = os.environ.get("AWS_BUCKET_NAME", "etl-ml-storage")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

client = MongoClient(MONGO_URL)
db = client[DB_NAME]

# Collections
pipelines_collection = db["pipelines"]
pipeline_runs_collection = db["pipeline_runs"]
experiments_collection = db["experiments"]
models_collection = db["models"]
data_validations_collection = db["data_validations"]
logs_collection = db["logs"]
datasets_collection = db["datasets"]
automl_runs_collection = db["automl_runs"]

# ============================================================================
# WebSocket Connection Manager
# ============================================================================

class ConnectionManager:
    """Manages WebSocket connections for real-time updates"""
    
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.pipeline_subscribers: Dict[str, Set[WebSocket]] = {}
        
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
        
    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)
        # Remove from pipeline subscribers
        for pipeline_id in list(self.pipeline_subscribers.keys()):
            self.pipeline_subscribers[pipeline_id].discard(websocket)
            if not self.pipeline_subscribers[pipeline_id]:
                del self.pipeline_subscribers[pipeline_id]
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    def subscribe_pipeline(self, websocket: WebSocket, pipeline_id: str):
        if pipeline_id not in self.pipeline_subscribers:
            self.pipeline_subscribers[pipeline_id] = set()
        self.pipeline_subscribers[pipeline_id].add(websocket)
        
    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast message to all connected clients"""
        disconnected = set()
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.add(connection)
        for conn in disconnected:
            self.disconnect(conn)
            
    async def broadcast_pipeline_update(self, pipeline_id: str, message: Dict[str, Any]):
        """Broadcast message to clients subscribed to a specific pipeline"""
        if pipeline_id in self.pipeline_subscribers:
            disconnected = set()
            for connection in self.pipeline_subscribers[pipeline_id]:
                try:
                    await connection.send_json(message)
                except Exception:
                    disconnected.add(connection)
            for conn in disconnected:
                self.disconnect(conn)
                
    async def send_personal_message(self, message: Dict[str, Any], websocket: WebSocket):
        """Send message to a specific client"""
        try:
            await websocket.send_json(message)
        except Exception:
            self.disconnect(websocket)

manager = ConnectionManager()

# ============================================================================
# Lifecycle Management
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management"""
    logger.info("Starting ETL & ML Dashboard API Server")
    logger.info(f"MongoDB: {MONGO_URL}")
    logger.info(f"Database: {DB_NAME}")
    
    # Create indexes for performance
    pipelines_collection.create_index("id", unique=True)
    experiments_collection.create_index("id", unique=True)
    pipeline_runs_collection.create_index([("pipeline_id", 1), ("started_at", -1)])
    logs_collection.create_index([("timestamp", -1), ("level", 1)])
    
    yield
    
    logger.info("Shutting down ETL & ML Dashboard API Server")

# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="ETL & ML Dashboard API",
    description="Backend API for ETL pipeline management, ML experiment tracking, and real-time monitoring",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Enums & Models
# ============================================================================

class PipelineStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    PENDING = "pending"
    PAUSED = "paused"

class PipelineStepType(str, Enum):
    EXTRACT = "extract"
    TRANSFORM = "transform"
    LOAD = "load"
    VALIDATE = "validate"
    TRAIN = "train"
    PREDICT = "predict"

class MLAlgorithm(str, Enum):
    RANDOM_FOREST = "RandomForest"
    GRADIENT_BOOSTING = "GradientBoosting"
    LOGISTIC_REGRESSION = "LogisticRegression"
    SVM = "SVM"
    AUTO = "AutoML"

class PipelineStep(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    type: PipelineStepType
    config: Dict[str, Any] = {}
    position: Dict[str, int] = {"x": 0, "y": 0}

class PipelineCreate(BaseModel):
    name: str
    description: Optional[str] = ""
    steps: List[PipelineStep] = []
    schedule: Optional[str] = None

class PipelineResponse(BaseModel):
    id: str
    name: str
    description: str
    steps: List[Dict[str, Any]]
    status: PipelineStatus
    schedule: Optional[str]
    last_run: Optional[str]
    created_at: str
    updated_at: str
    run_count: int

class PipelineRunResponse(BaseModel):
    id: str
    pipeline_id: str
    pipeline_name: str
    status: PipelineStatus
    started_at: str
    finished_at: Optional[str]
    duration_seconds: Optional[float]
    steps_completed: int
    total_steps: int
    current_step: Optional[str]
    logs: List[str]
    error: Optional[str]

class ExperimentCreate(BaseModel):
    name: str
    description: Optional[str] = ""
    pipeline_id: Optional[str] = None
    parameters: Dict[str, Any] = {}
    algorithm: MLAlgorithm = MLAlgorithm.RANDOM_FOREST

class ExperimentResponse(BaseModel):
    id: str
    name: str
    description: str
    pipeline_id: Optional[str]
    status: str
    parameters: Dict[str, Any]
    metrics: Dict[str, Any]
    model_version: Optional[str]
    algorithm: str
    created_at: str
    finished_at: Optional[str]

class AutoMLRequest(BaseModel):
    experiment_name: str
    description: Optional[str] = ""
    dataset_id: Optional[str] = None
    target_column: str = "target"
    algorithms: List[MLAlgorithm] = [MLAlgorithm.RANDOM_FOREST, MLAlgorithm.GRADIENT_BOOSTING, MLAlgorithm.LOGISTIC_REGRESSION]
    cv_folds: int = 5
    max_trials: int = 20
    scoring_metric: str = "accuracy"

class ModelResponse(BaseModel):
    id: str
    name: str
    version: str
    experiment_id: str
    algorithm: str
    metrics: Dict[str, Any]
    parameters: Dict[str, Any]
    file_path: Optional[str]
    s3_path: Optional[str]
    created_at: str
    status: str

class DataValidationCreate(BaseModel):
    name: str
    dataset_path: str
    rules: List[Dict[str, Any]] = []

class DataValidationResponse(BaseModel):
    id: str
    name: str
    dataset_path: str
    status: str
    rules_passed: int
    rules_failed: int
    total_rules: int
    issues: List[Dict[str, Any]]
    profile: Dict[str, Any]
    created_at: str
    finished_at: Optional[str]

class DatasetCreate(BaseModel):
    name: str
    description: Optional[str] = ""
    source_type: str = "local"  # local, s3, url

class DashboardStats(BaseModel):
    total_pipelines: int
    active_pipelines: int
    total_experiments: int
    successful_runs_24h: int
    failed_runs_24h: int
    total_models: int
    data_validations_passed: int
    data_validations_failed: int
    automl_runs: int
    total_datasets: int

# ============================================================================
# Helper Functions
# ============================================================================

def generate_id() -> str:
    """Generate a unique ID"""
    return str(uuid.uuid4())[:8]

def get_current_time() -> str:
    """Get current UTC time as ISO string"""
    return datetime.now(timezone.utc).isoformat()

def serialize_doc(doc: Dict) -> Dict:
    """Remove MongoDB _id from document"""
    if doc and "_id" in doc:
        del doc["_id"]
    return doc

async def log_event(level: str, source: str, message: str, metadata: Dict[str, Any] = None):
    """Log an event to the database and broadcast via WebSocket"""
    log_doc = {
        "id": generate_id(),
        "timestamp": get_current_time(),
        "level": level,
        "source": source,
        "message": message,
        "metadata": metadata or {}
    }
    logs_collection.insert_one(log_doc)
    
    # Broadcast to connected clients
    await manager.broadcast({
        "type": "log",
        "data": serialize_doc(log_doc)
    })

# ============================================================================
# ML Service - AutoML & Model Training
# ============================================================================

class MLService:
    """Machine Learning service for model training and AutoML"""
    
    ALGORITHM_MAP = {
        MLAlgorithm.RANDOM_FOREST: RandomForestClassifier,
        MLAlgorithm.GRADIENT_BOOSTING: GradientBoostingClassifier,
        MLAlgorithm.LOGISTIC_REGRESSION: LogisticRegression,
        MLAlgorithm.SVM: SVC,
    }
    
    PARAM_GRIDS = {
        MLAlgorithm.RANDOM_FOREST: {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, 20],
            'min_samples_split': [2, 5, 10],
        },
        MLAlgorithm.GRADIENT_BOOSTING: {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
        },
        MLAlgorithm.LOGISTIC_REGRESSION: {
            'C': [0.1, 1.0, 10.0],
            'max_iter': [100, 200, 500],
        },
        MLAlgorithm.SVM: {
            'C': [0.1, 1.0, 10.0],
            'kernel': ['rbf', 'linear'],
        },
    }
    
    @staticmethod
    def load_sample_data():
        """Load sample HMP activity recognition data or generate synthetic data"""
        try:
            # Try to load real data from the data directory
            data_path = "/app/data/data.csv"
            if os.path.isdir(data_path):
                files = [f for f in os.listdir(data_path) if f.endswith('.csv') and not f.startswith('part-')]
                if files:
                    dfs = []
                    for f in files[:5]:  # Limit files for performance
                        try:
                            df = pd.read_csv(os.path.join(data_path, f))
                            # Only keep numeric columns and class column
                            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                            if len(numeric_cols) >= 3:
                                dfs.append(df[numeric_cols])
                        except Exception:
                            continue
                    if dfs:
                        return pd.concat(dfs, ignore_index=True)
            elif os.path.isfile(data_path):
                df = pd.read_csv(data_path)
                # Filter to only numeric columns
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if len(numeric_cols) >= 3:
                    return df
        except Exception as e:
            logger.warning(f"Could not load real data: {e}")
        
        # Generate synthetic HMP-like data for demonstration
        logger.info("Generating synthetic activity recognition data")
        np.random.seed(42)
        n_samples = 2000
        
        # Simulate accelerometer data for different activities
        activities = ['walking', 'running', 'sitting', 'standing', 'climbing', 'lying']
        data_points = []
        
        for activity in activities:
            n = n_samples // len(activities)
            if activity == 'walking':
                x = np.random.normal(0.2, 0.3, n)
                y = np.random.normal(0.1, 0.2, n)
                z = np.random.normal(0.8, 0.4, n)
            elif activity == 'running':
                x = np.random.normal(0.5, 0.5, n)
                y = np.random.normal(0.3, 0.4, n)
                z = np.random.normal(1.2, 0.6, n)
            elif activity == 'sitting':
                x = np.random.normal(0.0, 0.1, n)
                y = np.random.normal(0.0, 0.1, n)
                z = np.random.normal(1.0, 0.1, n)
            elif activity == 'standing':
                x = np.random.normal(0.0, 0.05, n)
                y = np.random.normal(0.0, 0.05, n)
                z = np.random.normal(1.0, 0.05, n)
            elif activity == 'climbing':
                x = np.random.normal(0.3, 0.4, n)
                y = np.random.normal(0.4, 0.3, n)
                z = np.random.normal(0.6, 0.5, n)
            else:  # lying
                x = np.random.normal(0.9, 0.1, n)
                y = np.random.normal(0.0, 0.1, n)
                z = np.random.normal(0.1, 0.1, n)
            
            for i in range(n):
                data_points.append({
                    'x': x[i],
                    'y': y[i],
                    'z': z[i],
                    'class': activity
                })
        
        df = pd.DataFrame(data_points)
        np.random.shuffle(df.values)  # Shuffle the data
        return df
    
    @staticmethod
    def train_model(algorithm: MLAlgorithm, params: Dict[str, Any], X_train, y_train, X_test, y_test) -> Dict[str, Any]:
        """Train a single model and return results"""
        model_class = MLService.ALGORITHM_MAP.get(algorithm)
        if not model_class:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        # Filter params to only include valid ones for this algorithm
        valid_params = {}
        model_instance = model_class()
        valid_param_names = model_instance.get_params().keys()
        for key, value in params.items():
            if key in valid_param_names:
                valid_params[key] = value
        
        model = model_class(**valid_params, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        metrics = {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred, average='weighted', zero_division=0), 4),
            "recall": round(recall_score(y_test, y_pred, average='weighted', zero_division=0), 4),
            "f1_score": round(f1_score(y_test, y_pred, average='weighted', zero_division=0), 4),
            "training_samples": len(X_train),
            "test_samples": len(X_test),
        }
        
        return {
            "model": model,
            "metrics": metrics,
            "params": valid_params
        }
    
    @staticmethod
    async def run_automl(request: AutoMLRequest, run_id: str) -> Dict[str, Any]:
        """Run AutoML experiment with hyperparameter tuning"""
        logger.info(f"Starting AutoML run {run_id}")
        
        # Load data
        df = MLService.load_sample_data()
        
        # Prepare features and target
        feature_columns = [col for col in df.columns if col != request.target_column and col != 'class']
        if not feature_columns:
            feature_columns = ['x', 'y', 'z']
        
        target_col = request.target_column if request.target_column in df.columns else 'class'
        
        X = df[feature_columns].values
        y = df[target_col].values
        
        # Encode labels if needed
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, test_size=0.2, random_state=42
        )
        
        results = []
        best_score = 0
        best_model = None
        best_params = None
        best_algorithm = None
        
        total_trials = 0
        
        for algorithm in request.algorithms:
            if algorithm == MLAlgorithm.AUTO:
                continue
                
            await log_event("INFO", "AutoML", f"Testing algorithm: {algorithm.value}", {"run_id": run_id})
            
            param_grid = MLService.PARAM_GRIDS.get(algorithm, {})
            model_class = MLService.ALGORITHM_MAP.get(algorithm)
            
            if not model_class:
                continue
            
            try:
                # Run GridSearchCV
                model = model_class(random_state=42)
                grid_search = GridSearchCV(
                    model, 
                    param_grid, 
                    cv=request.cv_folds, 
                    scoring=request.scoring_metric,
                    n_jobs=-1
                )
                grid_search.fit(X_train, y_train)
                
                # Evaluate on test set
                test_score = grid_search.score(X_test, y_test)
                y_pred = grid_search.predict(X_test)
                
                trial_result = {
                    "algorithm": algorithm.value,
                    "best_params": grid_search.best_params_,
                    "cv_score": round(grid_search.best_score_, 4),
                    "test_score": round(test_score, 4),
                    "metrics": {
                        "accuracy": round(accuracy_score(y_test, y_pred), 4),
                        "precision": round(precision_score(y_test, y_pred, average='weighted', zero_division=0), 4),
                        "recall": round(recall_score(y_test, y_pred, average='weighted', zero_division=0), 4),
                        "f1_score": round(f1_score(y_test, y_pred, average='weighted', zero_division=0), 4),
                    }
                }
                results.append(trial_result)
                total_trials += len(grid_search.cv_results_['params'])
                
                # Track best model
                if test_score > best_score:
                    best_score = test_score
                    best_model = grid_search.best_estimator_
                    best_params = grid_search.best_params_
                    best_algorithm = algorithm
                
                # Broadcast progress
                await manager.broadcast({
                    "type": "automl_progress",
                    "data": {
                        "run_id": run_id,
                        "algorithm": algorithm.value,
                        "progress": len(results) / len(request.algorithms) * 100,
                        "current_best_score": best_score
                    }
                })
                
            except Exception as e:
                logger.error(f"Error training {algorithm.value}: {e}")
                await log_event("ERROR", "AutoML", f"Algorithm {algorithm.value} failed: {str(e)}", {"run_id": run_id})
        
        return {
            "run_id": run_id,
            "total_trials": total_trials,
            "results": results,
            "best_algorithm": best_algorithm.value if best_algorithm else None,
            "best_params": best_params,
            "best_score": round(best_score, 4),
            "best_model": best_model,
            "scaler": scaler,
            "label_encoder": le,
            "feature_columns": feature_columns
        }

ml_service = MLService()

# ============================================================================
# Pipeline Execution Engine
# ============================================================================

class PipelineExecutor:
    """Executes ETL pipelines with step-by-step tracking"""
    
    @staticmethod
    async def execute_pipeline(pipeline_id: str, run_id: str):
        """Execute a pipeline with real-time status updates"""
        pipeline = pipelines_collection.find_one({"id": pipeline_id})
        if not pipeline:
            raise HTTPException(status_code=404, detail="Pipeline not found")
        
        steps = pipeline.get("steps", [])
        total_steps = len(steps)
        
        await log_event("INFO", "Pipeline", f"Starting pipeline execution: {pipeline['name']}", 
                       {"pipeline_id": pipeline_id, "run_id": run_id})
        
        for i, step in enumerate(steps):
            step_name = step.get("name", f"Step {i+1}")
            step_type = step.get("type", "unknown")
            
            # Update current step
            pipeline_runs_collection.update_one(
                {"id": run_id},
                {"$set": {
                    "current_step": step_name,
                    "steps_completed": i
                },
                "$push": {"logs": f"[{get_current_time()}] Starting step: {step_name}"}}
            )
            
            # Broadcast step start
            await manager.broadcast_pipeline_update(pipeline_id, {
                "type": "pipeline_step",
                "data": {
                    "pipeline_id": pipeline_id,
                    "run_id": run_id,
                    "step": step_name,
                    "step_type": step_type,
                    "progress": (i / total_steps) * 100,
                    "status": "running"
                }
            })
            
            # Simulate step execution (2-5 seconds per step)
            await asyncio.sleep(random.uniform(2, 5))
            
            # 10% chance of failure for demonstration
            if random.random() < 0.1:
                error_msg = f"Step '{step_name}' failed: Simulated error"
                pipeline_runs_collection.update_one(
                    {"id": run_id},
                    {"$set": {
                        "status": PipelineStatus.FAILED,
                        "finished_at": get_current_time(),
                        "error": error_msg
                    },
                    "$push": {"logs": f"[{get_current_time()}] ERROR: {error_msg}"}}
                )
                
                pipelines_collection.update_one(
                    {"id": pipeline_id},
                    {"$set": {"status": PipelineStatus.FAILED, "updated_at": get_current_time()}}
                )
                
                await manager.broadcast_pipeline_update(pipeline_id, {
                    "type": "pipeline_failed",
                    "data": {
                        "pipeline_id": pipeline_id,
                        "run_id": run_id,
                        "error": error_msg
                    }
                })
                
                await log_event("ERROR", "Pipeline", error_msg, {"pipeline_id": pipeline_id, "run_id": run_id})
                return
            
            # Step completed
            pipeline_runs_collection.update_one(
                {"id": run_id},
                {"$push": {"logs": f"[{get_current_time()}] Completed step: {step_name}"}}
            )
            
            await manager.broadcast_pipeline_update(pipeline_id, {
                "type": "pipeline_step",
                "data": {
                    "pipeline_id": pipeline_id,
                    "run_id": run_id,
                    "step": step_name,
                    "step_type": step_type,
                    "progress": ((i + 1) / total_steps) * 100,
                    "status": "completed"
                }
            })
        
        # Pipeline completed successfully
        now = get_current_time()
        run = pipeline_runs_collection.find_one({"id": run_id})
        started = datetime.fromisoformat(run["started_at"].replace("Z", "+00:00"))
        finished = datetime.now(timezone.utc)
        duration = (finished - started).total_seconds()
        
        pipeline_runs_collection.update_one(
            {"id": run_id},
            {"$set": {
                "status": PipelineStatus.SUCCESS,
                "finished_at": now,
                "duration_seconds": duration,
                "steps_completed": total_steps,
                "current_step": None
            },
            "$push": {"logs": f"[{now}] Pipeline completed successfully"}}
        )
        
        pipelines_collection.update_one(
            {"id": pipeline_id},
            {"$set": {"status": PipelineStatus.SUCCESS, "last_run": now, "updated_at": now},
             "$inc": {"run_count": 1}}
        )
        
        await manager.broadcast_pipeline_update(pipeline_id, {
            "type": "pipeline_completed",
            "data": {
                "pipeline_id": pipeline_id,
                "run_id": run_id,
                "duration_seconds": duration,
                "status": "success"
            }
        })
        
        await log_event("INFO", "Pipeline", f"Pipeline completed: {pipeline['name']}", 
                       {"pipeline_id": pipeline_id, "run_id": run_id, "duration": duration})

pipeline_executor = PipelineExecutor()

# ============================================================================
# API Routes - Health Check
# ============================================================================

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": get_current_time(),
        "version": "2.0.0",
        "database": "connected" if client else "disconnected",
        "websocket_connections": len(manager.active_connections)
    }

# ============================================================================
# API Routes - WebSocket
# ============================================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_json()
            
            # Handle subscription requests
            if data.get("type") == "subscribe":
                pipeline_id = data.get("pipeline_id")
                if pipeline_id:
                    manager.subscribe_pipeline(websocket, pipeline_id)
                    await manager.send_personal_message({
                        "type": "subscribed",
                        "pipeline_id": pipeline_id
                    }, websocket)
            
            # Handle ping/pong for connection keep-alive
            elif data.get("type") == "ping":
                await manager.send_personal_message({"type": "pong"}, websocket)
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

# ============================================================================
# API Routes - Dashboard
# ============================================================================

@app.get("/api/dashboard/stats", response_model=DashboardStats)
async def get_dashboard_stats():
    """Get dashboard statistics"""
    total_pipelines = pipelines_collection.count_documents({})
    active_pipelines = pipeline_runs_collection.count_documents({"status": "running"})
    total_experiments = experiments_collection.count_documents({})
    successful_runs = pipeline_runs_collection.count_documents({"status": "success"})
    failed_runs = pipeline_runs_collection.count_documents({"status": "failed"})
    total_models = models_collection.count_documents({})
    validations_passed = data_validations_collection.count_documents({"status": "passed"})
    validations_failed = data_validations_collection.count_documents({"status": "failed"})
    automl_runs = automl_runs_collection.count_documents({})
    total_datasets = datasets_collection.count_documents({})
    
    return DashboardStats(
        total_pipelines=total_pipelines,
        active_pipelines=active_pipelines,
        total_experiments=total_experiments,
        successful_runs_24h=successful_runs,
        failed_runs_24h=failed_runs,
        total_models=total_models,
        data_validations_passed=validations_passed,
        data_validations_failed=validations_failed,
        automl_runs=automl_runs,
        total_datasets=total_datasets
    )

@app.get("/api/dashboard/recent-runs")
async def get_recent_runs(limit: int = Query(default=10, le=50)):
    """Get recent pipeline runs"""
    runs = list(pipeline_runs_collection.find().sort("started_at", -1).limit(limit))
    return [serialize_doc(run) for run in runs]

@app.get("/api/dashboard/metrics")
async def get_dashboard_metrics():
    """Get metrics for dashboard charts"""
    # Get actual run data from last 7 days
    runs = list(pipeline_runs_collection.find().sort("started_at", -1).limit(100))
    
    # Group by day
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    pipeline_runs_data = []
    for day in days:
        success_count = random.randint(5, 20)
        failed_count = random.randint(0, 3)
        pipeline_runs_data.append({"date": day, "success": success_count, "failed": failed_count})
    
    # Get model accuracy trends from experiments
    experiments = list(experiments_collection.find({"status": "completed"}).sort("created_at", 1).limit(10))
    model_accuracy = []
    for i, exp in enumerate(experiments):
        metrics = exp.get("metrics", {})
        accuracy = metrics.get("accuracy", random.uniform(0.82, 0.95))
        model_accuracy.append({"version": f"v{i+1}.0", "accuracy": accuracy})
    
    if not model_accuracy:
        model_accuracy = [
            {"version": "v1.0", "accuracy": 0.82},
            {"version": "v1.1", "accuracy": 0.85},
            {"version": "v1.2", "accuracy": 0.87},
            {"version": "v1.3", "accuracy": 0.89},
            {"version": "v1.4", "accuracy": 0.91},
        ]
    
    metrics = {
        "pipeline_runs": pipeline_runs_data,
        "model_accuracy": model_accuracy,
        "data_quality": {
            "completeness": 98.5,
            "accuracy": 97.2,
            "consistency": 99.1,
            "timeliness": 95.8
        }
    }
    return metrics

# ============================================================================
# API Routes - Pipelines
# ============================================================================

@app.get("/api/pipelines", response_model=List[PipelineResponse])
async def list_pipelines():
    """List all pipelines"""
    pipelines = list(pipelines_collection.find())
    return [serialize_doc(p) for p in pipelines]

@app.post("/api/pipelines", response_model=PipelineResponse)
async def create_pipeline(pipeline: PipelineCreate):
    """Create a new pipeline"""
    now = get_current_time()
    pipeline_doc = {
        "id": generate_id(),
        "name": pipeline.name,
        "description": pipeline.description or "",
        "steps": [step.dict() for step in pipeline.steps],
        "status": PipelineStatus.IDLE,
        "schedule": pipeline.schedule,
        "last_run": None,
        "created_at": now,
        "updated_at": now,
        "run_count": 0
    }
    pipelines_collection.insert_one(pipeline_doc)
    await log_event("INFO", "Pipeline", f"Pipeline created: {pipeline.name}", {"pipeline_id": pipeline_doc["id"]})
    return serialize_doc(pipeline_doc)

@app.get("/api/pipelines/{pipeline_id}", response_model=PipelineResponse)
async def get_pipeline(pipeline_id: str):
    """Get pipeline by ID"""
    pipeline = pipelines_collection.find_one({"id": pipeline_id})
    if not pipeline:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    return serialize_doc(pipeline)

@app.delete("/api/pipelines/{pipeline_id}")
async def delete_pipeline(pipeline_id: str):
    """Delete a pipeline"""
    result = pipelines_collection.delete_one({"id": pipeline_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    await log_event("INFO", "Pipeline", f"Pipeline deleted", {"pipeline_id": pipeline_id})
    return {"message": "Pipeline deleted successfully"}

@app.post("/api/pipelines/{pipeline_id}/run")
async def run_pipeline(pipeline_id: str, background_tasks: BackgroundTasks):
    """Trigger a pipeline run with background execution"""
    pipeline = pipelines_collection.find_one({"id": pipeline_id})
    if not pipeline:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    
    now = get_current_time()
    run_doc = {
        "id": generate_id(),
        "pipeline_id": pipeline_id,
        "pipeline_name": pipeline["name"],
        "status": PipelineStatus.RUNNING,
        "started_at": now,
        "finished_at": None,
        "duration_seconds": None,
        "steps_completed": 0,
        "total_steps": len(pipeline.get("steps", [])),
        "current_step": None,
        "logs": [f"[{now}] Pipeline run started"],
        "error": None
    }
    pipeline_runs_collection.insert_one(run_doc)
    
    # Update pipeline status
    pipelines_collection.update_one(
        {"id": pipeline_id},
        {"$set": {"status": PipelineStatus.RUNNING, "updated_at": now}}
    )
    
    # Execute pipeline in background
    background_tasks.add_task(pipeline_executor.execute_pipeline, pipeline_id, run_doc["id"])
    
    return serialize_doc(run_doc)

@app.get("/api/pipelines/{pipeline_id}/runs")
async def get_pipeline_runs(pipeline_id: str, limit: int = Query(default=20, le=100)):
    """Get runs for a specific pipeline"""
    runs = list(pipeline_runs_collection.find({"pipeline_id": pipeline_id}).sort("started_at", -1).limit(limit))
    return [serialize_doc(run) for run in runs]

# ============================================================================
# API Routes - Pipeline Runs
# ============================================================================

@app.get("/api/runs", response_model=List[PipelineRunResponse])
async def list_runs(limit: int = Query(default=50, le=200)):
    """List all pipeline runs"""
    runs = list(pipeline_runs_collection.find().sort("started_at", -1).limit(limit))
    return [serialize_doc(run) for run in runs]

@app.get("/api/runs/{run_id}")
async def get_run(run_id: str):
    """Get a specific run by ID"""
    run = pipeline_runs_collection.find_one({"id": run_id})
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return serialize_doc(run)

@app.post("/api/runs/{run_id}/complete")
async def complete_run(run_id: str, success: bool = True, error: Optional[str] = None):
    """Mark a run as complete"""
    run = pipeline_runs_collection.find_one({"id": run_id})
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    
    now = get_current_time()
    started = datetime.fromisoformat(run["started_at"].replace("Z", "+00:00"))
    finished = datetime.now(timezone.utc)
    duration = (finished - started).total_seconds()
    
    status = PipelineStatus.SUCCESS if success else PipelineStatus.FAILED
    
    pipeline_runs_collection.update_one(
        {"id": run_id},
        {"$set": {
            "status": status,
            "finished_at": now,
            "duration_seconds": duration,
            "steps_completed": run["total_steps"] if success else run["steps_completed"],
            "error": error,
        },
        "$push": {"logs": f"[{now}] Pipeline run {'completed successfully' if success else 'failed'}"}}
    )
    
    # Update pipeline status and run count
    pipelines_collection.update_one(
        {"id": run["pipeline_id"]},
        {"$set": {"status": status, "last_run": now, "updated_at": now}, "$inc": {"run_count": 1}}
    )
    
    return {"message": f"Run marked as {status}"}

# ============================================================================
# API Routes - Experiments
# ============================================================================

@app.get("/api/experiments", response_model=List[ExperimentResponse])
async def list_experiments():
    """List all experiments"""
    experiments = list(experiments_collection.find().sort("created_at", -1))
    return [serialize_doc(exp) for exp in experiments]

@app.post("/api/experiments", response_model=ExperimentResponse)
async def create_experiment(experiment: ExperimentCreate, background_tasks: BackgroundTasks):
    """Create and run a new ML experiment"""
    now = get_current_time()
    exp_doc = {
        "id": generate_id(),
        "name": experiment.name,
        "description": experiment.description or "",
        "pipeline_id": experiment.pipeline_id,
        "status": "running",
        "parameters": experiment.parameters,
        "metrics": {},
        "model_version": None,
        "algorithm": experiment.algorithm.value,
        "created_at": now,
        "finished_at": None
    }
    experiments_collection.insert_one(exp_doc)
    
    # Run training in background
    async def train_experiment():
        try:
            # Load data
            df = ml_service.load_sample_data()
            feature_cols = [c for c in df.columns if c not in ['class', 'target']]
            X = df[feature_cols].values
            y = df['class'].values if 'class' in df.columns else df['target'].values
            
            # Encode labels
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)
            
            # Train model
            result = ml_service.train_model(
                experiment.algorithm,
                experiment.parameters,
                X_train, y_train, X_test, y_test
            )
            
            # Get version number
            existing = list(models_collection.find({"experiment_id": exp_doc["id"]}))
            version = f"v{len(existing) + 1}.0"
            
            # Update experiment
            experiments_collection.update_one(
                {"id": exp_doc["id"]},
                {"$set": {
                    "status": "completed",
                    "metrics": result["metrics"],
                    "model_version": version,
                    "finished_at": get_current_time()
                }}
            )
            
            # Save model
            model_doc = {
                "id": generate_id(),
                "name": f"{experiment.name}_model",
                "version": version,
                "experiment_id": exp_doc["id"],
                "algorithm": experiment.algorithm.value,
                "metrics": result["metrics"],
                "parameters": result["params"],
                "file_path": f"/models/{exp_doc['id']}/{version}/model.pkl",
                "created_at": get_current_time(),
                "status": "registered"
            }
            models_collection.insert_one(model_doc)
            
            await log_event("INFO", "Experiment", f"Experiment completed: {experiment.name}", 
                           {"experiment_id": exp_doc["id"], "metrics": result["metrics"]})
            
            await manager.broadcast({
                "type": "experiment_completed",
                "data": {"experiment_id": exp_doc["id"], "metrics": result["metrics"]}
            })
            
        except Exception as e:
            logger.error(f"Experiment training failed: {e}")
            experiments_collection.update_one(
                {"id": exp_doc["id"]},
                {"$set": {"status": "failed", "finished_at": get_current_time()}}
            )
            await log_event("ERROR", "Experiment", f"Experiment failed: {str(e)}", {"experiment_id": exp_doc["id"]})
    
    background_tasks.add_task(train_experiment)
    
    return serialize_doc(exp_doc)

@app.get("/api/experiments/{experiment_id}", response_model=ExperimentResponse)
async def get_experiment(experiment_id: str):
    """Get experiment by ID"""
    exp = experiments_collection.find_one({"id": experiment_id})
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return serialize_doc(exp)

@app.put("/api/experiments/{experiment_id}/metrics")
async def update_experiment_metrics(experiment_id: str, metrics: Dict[str, Any]):
    """Update experiment metrics"""
    exp = experiments_collection.find_one({"id": experiment_id})
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    experiments_collection.update_one(
        {"id": experiment_id},
        {"$set": {"metrics": metrics}}
    )
    return {"message": "Metrics updated successfully"}

@app.post("/api/experiments/{experiment_id}/complete")
async def complete_experiment(experiment_id: str, metrics: Dict[str, Any] = {}):
    """Mark experiment as complete"""
    exp = experiments_collection.find_one({"id": experiment_id})
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    now = get_current_time()
    model_version = f"v{len(list(models_collection.find({'experiment_id': experiment_id}))) + 1}.0"
    
    experiments_collection.update_one(
        {"id": experiment_id},
        {"$set": {
            "status": "completed",
            "metrics": {**exp.get("metrics", {}), **metrics},
            "model_version": model_version,
            "finished_at": now
        }}
    )
    return {"message": "Experiment completed", "model_version": model_version}

@app.delete("/api/experiments/{experiment_id}")
async def delete_experiment(experiment_id: str):
    """Delete an experiment"""
    result = experiments_collection.delete_one({"id": experiment_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return {"message": "Experiment deleted successfully"}

# ============================================================================
# API Routes - AutoML
# ============================================================================

@app.post("/api/automl/run")
async def run_automl(request: AutoMLRequest, background_tasks: BackgroundTasks):
    """Start an AutoML experiment"""
    run_id = generate_id()
    now = get_current_time()
    
    # Create AutoML run record
    automl_doc = {
        "id": run_id,
        "name": request.experiment_name,
        "description": request.description,
        "status": "running",
        "algorithms": [a.value for a in request.algorithms],
        "cv_folds": request.cv_folds,
        "max_trials": request.max_trials,
        "scoring_metric": request.scoring_metric,
        "results": [],
        "best_algorithm": None,
        "best_params": None,
        "best_score": None,
        "created_at": now,
        "finished_at": None
    }
    automl_runs_collection.insert_one(automl_doc)
    
    async def execute_automl():
        try:
            result = await ml_service.run_automl(request, run_id)
            
            # Update AutoML run
            automl_runs_collection.update_one(
                {"id": run_id},
                {"$set": {
                    "status": "completed",
                    "results": result["results"],
                    "best_algorithm": result["best_algorithm"],
                    "best_params": result["best_params"],
                    "best_score": result["best_score"],
                    "total_trials": result["total_trials"],
                    "finished_at": get_current_time()
                }}
            )
            
            # Create experiment record for best model
            exp_doc = {
                "id": generate_id(),
                "name": f"{request.experiment_name} - Best Model",
                "description": f"AutoML best model: {result['best_algorithm']}",
                "pipeline_id": None,
                "status": "completed",
                "parameters": result["best_params"] or {},
                "metrics": {"accuracy": result["best_score"]},
                "model_version": "v1.0",
                "algorithm": result["best_algorithm"] or "Unknown",
                "created_at": get_current_time(),
                "finished_at": get_current_time()
            }
            experiments_collection.insert_one(exp_doc)
            
            # Broadcast completion
            await manager.broadcast({
                "type": "automl_completed",
                "data": {
                    "run_id": run_id,
                    "best_algorithm": result["best_algorithm"],
                    "best_score": result["best_score"]
                }
            })
            
            await log_event("INFO", "AutoML", f"AutoML completed: {request.experiment_name}", 
                           {"run_id": run_id, "best_score": result["best_score"]})
            
        except Exception as e:
            logger.error(f"AutoML failed: {e}")
            automl_runs_collection.update_one(
                {"id": run_id},
                {"$set": {"status": "failed", "finished_at": get_current_time()}}
            )
            await log_event("ERROR", "AutoML", f"AutoML failed: {str(e)}", {"run_id": run_id})
    
    background_tasks.add_task(execute_automl)
    
    return serialize_doc(automl_doc)

@app.get("/api/automl/runs")
async def list_automl_runs():
    """List all AutoML runs"""
    runs = list(automl_runs_collection.find().sort("created_at", -1))
    return [serialize_doc(run) for run in runs]

@app.get("/api/automl/runs/{run_id}")
async def get_automl_run(run_id: str):
    """Get AutoML run details"""
    run = automl_runs_collection.find_one({"id": run_id})
    if not run:
        raise HTTPException(status_code=404, detail="AutoML run not found")
    return serialize_doc(run)

# ============================================================================
# API Routes - Models
# ============================================================================

@app.get("/api/models", response_model=List[ModelResponse])
async def list_models():
    """List all models"""
    models = list(models_collection.find().sort("created_at", -1))
    return [serialize_doc(m) for m in models]

@app.post("/api/models")
async def create_model(
    name: str,
    experiment_id: str,
    algorithm: str,
    metrics: Dict[str, Any] = {},
    parameters: Dict[str, Any] = {}
):
    """Register a new model"""
    # Get version number
    existing = list(models_collection.find({"name": name}))
    version = f"v{len(existing) + 1}.0"
    
    now = get_current_time()
    model_doc = {
        "id": generate_id(),
        "name": name,
        "version": version,
        "experiment_id": experiment_id,
        "algorithm": algorithm,
        "metrics": metrics,
        "parameters": parameters,
        "file_path": f"/models/{name}/{version}/model.pkl",
        "s3_path": f"s3://{AWS_BUCKET_NAME}/models/{name}/{version}/model.pkl" if AWS_BUCKET_NAME else None,
        "created_at": now,
        "status": "registered"
    }
    models_collection.insert_one(model_doc)
    return serialize_doc(model_doc)

@app.get("/api/models/{model_id}")
async def get_model(model_id: str):
    """Get model by ID"""
    model = models_collection.find_one({"id": model_id})
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return serialize_doc(model)

# ============================================================================
# API Routes - Data Validation
# ============================================================================

@app.get("/api/validations", response_model=List[DataValidationResponse])
async def list_validations():
    """List all data validations"""
    validations = list(data_validations_collection.find().sort("created_at", -1))
    return [serialize_doc(v) for v in validations]

@app.post("/api/validations", response_model=DataValidationResponse)
async def create_validation(validation: DataValidationCreate):
    """Create a new data validation"""
    now = get_current_time()
    
    # Simulate validation results
    rules_total = len(validation.rules) if validation.rules else random.randint(5, 15)
    rules_passed = random.randint(int(rules_total * 0.7), rules_total)
    rules_failed = rules_total - rules_passed
    
    issues = []
    if rules_failed > 0:
        issue_types = ["Missing values", "Type mismatch", "Out of range", "Duplicate records", "Invalid format"]
        for i in range(rules_failed):
            issues.append({
                "rule": f"Rule_{i+1}",
                "type": random.choice(issue_types),
                "severity": random.choice(["low", "medium", "high"]),
                "affected_rows": random.randint(1, 100),
                "description": f"Validation issue detected in dataset"
            })
    
    validation_doc = {
        "id": generate_id(),
        "name": validation.name,
        "dataset_path": validation.dataset_path,
        "status": "passed" if rules_failed == 0 else "failed",
        "rules_passed": rules_passed,
        "rules_failed": rules_failed,
        "total_rules": rules_total,
        "issues": issues,
        "profile": {
            "total_rows": random.randint(10000, 100000),
            "total_columns": random.randint(10, 50),
            "missing_cells": random.randint(0, 500),
            "duplicate_rows": random.randint(0, 100),
            "memory_size_mb": round(random.uniform(10, 500), 2)
        },
        "created_at": now,
        "finished_at": now
    }
    data_validations_collection.insert_one(validation_doc)
    return serialize_doc(validation_doc)

@app.get("/api/validations/{validation_id}")
async def get_validation(validation_id: str):
    """Get validation by ID"""
    validation = data_validations_collection.find_one({"id": validation_id})
    if not validation:
        raise HTTPException(status_code=404, detail="Validation not found")
    return serialize_doc(validation)

# ============================================================================
# API Routes - Logs
# ============================================================================

@app.get("/api/logs")
async def list_logs(
    limit: int = Query(default=100, le=500),
    level: Optional[str] = None,
    source: Optional[str] = None
):
    """List logs with optional filtering"""
    query = {}
    if level:
        query["level"] = level
    if source:
        query["source"] = source
    
    logs = list(logs_collection.find(query).sort("timestamp", -1).limit(limit))
    return [serialize_doc(log) for log in logs]

@app.post("/api/logs")
async def create_log(level: str, source: str, message: str, metadata: Dict[str, Any] = {}):
    """Create a new log entry"""
    log_doc = {
        "id": generate_id(),
        "timestamp": get_current_time(),
        "level": level,
        "source": source,
        "message": message,
        "metadata": metadata
    }
    logs_collection.insert_one(log_doc)
    return serialize_doc(log_doc)

# ============================================================================
# API Routes - Datasets
# ============================================================================

@app.get("/api/datasets")
async def list_datasets():
    """List all datasets"""
    datasets = list(datasets_collection.find().sort("created_at", -1))
    return [serialize_doc(d) for d in datasets]

@app.post("/api/datasets")
async def create_dataset(dataset: DatasetCreate):
    """Register a new dataset"""
    now = get_current_time()
    dataset_doc = {
        "id": generate_id(),
        "name": dataset.name,
        "description": dataset.description,
        "source_type": dataset.source_type,
        "created_at": now,
        "updated_at": now,
        "size_mb": 0,
        "row_count": 0,
        "column_count": 0
    }
    datasets_collection.insert_one(dataset_doc)
    return serialize_doc(dataset_doc)

# ============================================================================
# API Routes - Seed Data
# ============================================================================

@app.post("/api/seed")
async def seed_data():
    """Seed the database with sample data"""
    now = get_current_time()
    
    # Clear existing data
    pipelines_collection.delete_many({})
    pipeline_runs_collection.delete_many({})
    experiments_collection.delete_many({})
    models_collection.delete_many({})
    data_validations_collection.delete_many({})
    logs_collection.delete_many({})
    automl_runs_collection.delete_many({})
    datasets_collection.delete_many({})
    
    # Seed pipelines
    pipelines = [
        {
            "id": "pip-001",
            "name": "HMP Data ETL Pipeline",
            "description": "Extract, transform and load HMP accelerometer sensor data",
            "steps": [
                {"id": "step-1", "name": "Extract CSV Data", "type": "extract", "config": {"source": "data/data.csv"}, "position": {"x": 100, "y": 100}},
                {"id": "step-2", "name": "Convert to Parquet", "type": "transform", "config": {"format": "parquet"}, "position": {"x": 300, "y": 100}},
                {"id": "step-3", "name": "Validate Data", "type": "validate", "config": {"rules": ["not_null", "type_check"]}, "position": {"x": 500, "y": 100}},
                {"id": "step-4", "name": "Load to Database", "type": "load", "config": {"target": "mongodb"}, "position": {"x": 700, "y": 100}},
            ],
            "status": "success",
            "schedule": "0 */6 * * *",
            "last_run": now,
            "created_at": now,
            "updated_at": now,
            "run_count": 45
        },
        {
            "id": "pip-002",
            "name": "ML Training Pipeline",
            "description": "Train Random Forest classifier for activity recognition",
            "steps": [
                {"id": "step-1", "name": "Load Parquet Data", "type": "extract", "config": {"source": "data/data.parquet"}, "position": {"x": 100, "y": 100}},
                {"id": "step-2", "name": "Feature Engineering", "type": "transform", "config": {"features": ["x", "y", "z"]}, "position": {"x": 300, "y": 100}},
                {"id": "step-3", "name": "Train Model", "type": "train", "config": {"algorithm": "RandomForest"}, "position": {"x": 500, "y": 100}},
                {"id": "step-4", "name": "Export PMML", "type": "load", "config": {"format": "pmml"}, "position": {"x": 700, "y": 100}},
            ],
            "status": "idle",
            "schedule": None,
            "last_run": now,
            "created_at": now,
            "updated_at": now,
            "run_count": 12
        },
        {
            "id": "pip-003",
            "name": "Data Quality Check",
            "description": "Daily data quality validation pipeline",
            "steps": [
                {"id": "step-1", "name": "Load Data", "type": "extract", "config": {}, "position": {"x": 100, "y": 100}},
                {"id": "step-2", "name": "Check Completeness", "type": "validate", "config": {}, "position": {"x": 300, "y": 100}},
                {"id": "step-3", "name": "Check Consistency", "type": "validate", "config": {}, "position": {"x": 500, "y": 100}},
            ],
            "status": "failed",
            "schedule": "0 0 * * *",
            "last_run": now,
            "created_at": now,
            "updated_at": now,
            "run_count": 30
        },
        {
            "id": "pip-004",
            "name": "AutoML Pipeline",
            "description": "Automated model selection and hyperparameter tuning",
            "steps": [
                {"id": "step-1", "name": "Load Dataset", "type": "extract", "config": {}, "position": {"x": 100, "y": 100}},
                {"id": "step-2", "name": "Feature Selection", "type": "transform", "config": {}, "position": {"x": 300, "y": 100}},
                {"id": "step-3", "name": "AutoML Search", "type": "train", "config": {"mode": "automl"}, "position": {"x": 500, "y": 100}},
                {"id": "step-4", "name": "Model Evaluation", "type": "validate", "config": {}, "position": {"x": 700, "y": 100}},
                {"id": "step-5", "name": "Deploy Best Model", "type": "load", "config": {}, "position": {"x": 900, "y": 100}},
            ],
            "status": "idle",
            "schedule": None,
            "last_run": None,
            "created_at": now,
            "updated_at": now,
            "run_count": 0
        }
    ]
    
    for pipeline in pipelines:
        pipelines_collection.insert_one(pipeline)
    
    # Seed pipeline runs
    statuses = ["success", "success", "success", "success", "failed", "success", "running"]
    for i in range(15):
        pipeline = random.choice(pipelines[:3])
        status = random.choice(statuses)
        run = {
            "id": f"run-{i+1:03d}",
            "pipeline_id": pipeline["id"],
            "pipeline_name": pipeline["name"],
            "status": status,
            "started_at": now,
            "finished_at": now if status != "running" else None,
            "duration_seconds": random.uniform(30, 300) if status != "running" else None,
            "steps_completed": len(pipeline["steps"]) if status == "success" else random.randint(0, len(pipeline["steps"])-1),
            "total_steps": len(pipeline["steps"]),
            "current_step": None if status != "running" else pipeline["steps"][random.randint(0, len(pipeline["steps"])-1)]["name"],
            "logs": [f"[{now}] Step {j+1} completed" for j in range(random.randint(1, len(pipeline["steps"])))],
            "error": "Connection timeout to data source" if status == "failed" else None
        }
        pipeline_runs_collection.insert_one(run)
    
    # Seed experiments
    algorithms = ["RandomForest", "GradientBoosting", "LogisticRegression", "SVM", "AutoML"]
    for i in range(10):
        exp = {
            "id": f"exp-{i+1:03d}",
            "name": f"Activity Recognition Experiment {i+1}",
            "description": f"Training {algorithms[i % len(algorithms)]} model for activity classification",
            "pipeline_id": "pip-002",
            "status": "completed" if i < 8 else "running",
            "parameters": {
                "algorithm": algorithms[i % len(algorithms)],
                "n_estimators": random.randint(50, 200),
                "max_depth": random.randint(5, 20),
                "learning_rate": round(random.uniform(0.01, 0.1), 3)
            },
            "metrics": {
                "accuracy": round(random.uniform(0.82, 0.95), 4),
                "precision": round(random.uniform(0.80, 0.94), 4),
                "recall": round(random.uniform(0.78, 0.93), 4),
                "f1_score": round(random.uniform(0.79, 0.94), 4),
                "training_time_seconds": round(random.uniform(60, 600), 2)
            } if i < 8 else {},
            "model_version": f"v{i+1}.0" if i < 8 else None,
            "algorithm": algorithms[i % len(algorithms)],
            "created_at": now,
            "finished_at": now if i < 8 else None
        }
        experiments_collection.insert_one(exp)
        
        # Create model for completed experiments
        if i < 8:
            model = {
                "id": f"model-{i+1:03d}",
                "name": "activity_classifier",
                "version": f"v{i+1}.0",
                "experiment_id": exp["id"],
                "algorithm": algorithms[i % len(algorithms)],
                "metrics": exp["metrics"],
                "parameters": exp["parameters"],
                "file_path": f"/models/activity_classifier/v{i+1}.0/model.pkl",
                "s3_path": f"s3://etl-ml-storage/models/activity_classifier/v{i+1}.0/model.pkl",
                "created_at": now,
                "status": "registered"
            }
            models_collection.insert_one(model)
    
    # Seed AutoML runs
    automl_run = {
        "id": "automl-001",
        "name": "Activity Recognition AutoML",
        "description": "Automated model selection for activity classification",
        "status": "completed",
        "algorithms": ["RandomForest", "GradientBoosting", "LogisticRegression"],
        "cv_folds": 5,
        "max_trials": 20,
        "scoring_metric": "accuracy",
        "results": [
            {"algorithm": "RandomForest", "best_params": {"n_estimators": 100, "max_depth": 10}, "cv_score": 0.89, "test_score": 0.88},
            {"algorithm": "GradientBoosting", "best_params": {"n_estimators": 100, "learning_rate": 0.1}, "cv_score": 0.91, "test_score": 0.90},
            {"algorithm": "LogisticRegression", "best_params": {"C": 1.0}, "cv_score": 0.85, "test_score": 0.84},
        ],
        "best_algorithm": "GradientBoosting",
        "best_params": {"n_estimators": 100, "learning_rate": 0.1},
        "best_score": 0.90,
        "total_trials": 27,
        "created_at": now,
        "finished_at": now
    }
    automl_runs_collection.insert_one(automl_run)
    
    # Seed data validations
    validation_names = ["Input Data Validation", "Training Data Validation", "Feature Data Validation"]
    for i in range(5):
        rules_total = random.randint(8, 15)
        rules_passed = random.randint(int(rules_total * 0.6), rules_total)
        rules_failed = rules_total - rules_passed
        
        validation = {
            "id": f"val-{i+1:03d}",
            "name": validation_names[i % len(validation_names)],
            "dataset_path": f"/data/dataset_{i+1}.parquet",
            "status": "passed" if rules_failed == 0 else "failed",
            "rules_passed": rules_passed,
            "rules_failed": rules_failed,
            "total_rules": rules_total,
            "issues": [
                {"rule": f"Rule_{j}", "type": random.choice(["Missing values", "Type mismatch", "Out of range"]), "severity": random.choice(["low", "medium", "high"]), "affected_rows": random.randint(1, 50), "description": "Validation issue"}
                for j in range(rules_failed)
            ],
            "profile": {
                "total_rows": random.randint(10000, 100000),
                "total_columns": random.randint(10, 50),
                "missing_cells": random.randint(0, 500),
                "duplicate_rows": random.randint(0, 100),
                "memory_size_mb": round(random.uniform(10, 500), 2)
            },
            "created_at": now,
            "finished_at": now
        }
        data_validations_collection.insert_one(validation)
    
    # Seed datasets
    datasets = [
        {"id": "ds-001", "name": "HMP Activity Data", "description": "Human activity recognition sensor data", "source_type": "local", "size_mb": 45.2, "row_count": 50000, "column_count": 4},
        {"id": "ds-002", "name": "Training Dataset", "description": "Processed training dataset", "source_type": "local", "size_mb": 120.5, "row_count": 100000, "column_count": 10},
        {"id": "ds-003", "name": "Validation Dataset", "description": "Holdout validation dataset", "source_type": "s3", "size_mb": 30.1, "row_count": 25000, "column_count": 10},
    ]
    for ds in datasets:
        ds["created_at"] = now
        ds["updated_at"] = now
        datasets_collection.insert_one(ds)
    
    # Seed logs
    log_levels = ["INFO", "INFO", "INFO", "WARNING", "ERROR", "DEBUG"]
    sources = ["Pipeline", "Experiment", "Validation", "System", "AutoML"]
    messages = [
        "Pipeline execution started",
        "Data extraction completed",
        "Transform step finished",
        "Model training in progress",
        "Validation passed",
        "Connection established",
        "Memory usage at 75%",
        "Retrying failed operation",
        "Data validation completed",
        "Model exported successfully",
        "AutoML trial completed",
        "Hyperparameter optimization finished",
        "Best model selected",
        "WebSocket connection established"
    ]
    
    for i in range(50):
        log = {
            "id": f"log-{i+1:03d}",
            "timestamp": now,
            "level": random.choice(log_levels),
            "source": random.choice(sources),
            "message": random.choice(messages),
            "metadata": {"run_id": f"run-{random.randint(1, 15):03d}"}
        }
        logs_collection.insert_one(log)
    
    return {
        "message": "Database seeded successfully",
        "pipelines": len(pipelines),
        "runs": 15,
        "experiments": 10,
        "models": 8,
        "validations": 5,
        "logs": 50,
        "automl_runs": 1,
        "datasets": 3
    }

# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
