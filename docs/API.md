# API Reference

Complete API documentation for the ETL & ML Platform.

## Base URL

```
http://localhost:8001/api
```

## Authentication

Currently, the API does not require authentication. For production deployments, implement JWT-based authentication.

## Response Format

All responses are JSON with the following structure:

**Success Response:**
```json
{
  "data": { ... },
  "message": "Success message"
}
```

**Error Response:**
```json
{
  "detail": "Error description"
}
```

## HTTP Status Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 201 | Created |
| 400 | Bad Request |
| 404 | Not Found |
| 422 | Validation Error |
| 500 | Internal Server Error |

---

## Health Check

### GET /api/health

Check the health status of the API.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "1.0.0",
  "database": "connected"
}
```

---

## Dashboard

### GET /api/dashboard/stats

Get overall statistics for the dashboard.

**Response:**
```json
{
  "total_pipelines": 3,
  "active_pipelines": 1,
  "total_experiments": 10,
  "successful_runs_24h": 45,
  "failed_runs_24h": 5,
  "total_models": 8,
  "data_validations_passed": 12,
  "data_validations_failed": 3
}
```

### GET /api/dashboard/metrics

Get metrics data for charts.

**Response:**
```json
{
  "pipeline_runs": [
    {"date": "Mon", "success": 12, "failed": 2},
    {"date": "Tue", "success": 15, "failed": 1}
  ],
  "model_accuracy": [
    {"version": "v1.0", "accuracy": 0.82},
    {"version": "v1.1", "accuracy": 0.85}
  ],
  "data_quality": {
    "completeness": 98.5,
    "accuracy": 97.2,
    "consistency": 99.1,
    "timeliness": 95.8
  }
}
```

### GET /api/dashboard/recent-runs

Get recent pipeline runs.

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| limit | integer | 10 | Maximum number of runs (max: 50) |

**Response:**
```json
[
  {
    "id": "run-001",
    "pipeline_id": "pip-001",
    "pipeline_name": "Data ETL Pipeline",
    "status": "success",
    "started_at": "2024-01-15T10:00:00Z",
    "finished_at": "2024-01-15T10:05:00Z",
    "duration_seconds": 300.5,
    "steps_completed": 4,
    "total_steps": 4
  }
]
```

---

## Pipelines

### GET /api/pipelines

List all pipelines.

**Response:**
```json
[
  {
    "id": "pip-001",
    "name": "HMP Data ETL Pipeline",
    "description": "Extract, transform and load HMP data",
    "steps": [
      {
        "id": "step-1",
        "name": "Extract CSV Data",
        "type": "extract",
        "config": {"source": "data/data.csv"},
        "position": {"x": 100, "y": 100}
      }
    ],
    "status": "success",
    "schedule": "0 */6 * * *",
    "last_run": "2024-01-15T10:00:00Z",
    "created_at": "2024-01-01T00:00:00Z",
    "updated_at": "2024-01-15T10:05:00Z",
    "run_count": 45
  }
]
```

### POST /api/pipelines

Create a new pipeline.

**Request Body:**
```json
{
  "name": "New ETL Pipeline",
  "description": "Pipeline description",
  "steps": [
    {
      "name": "Extract Data",
      "type": "extract",
      "config": {"source": "/data/input.csv"}
    },
    {
      "name": "Transform Data",
      "type": "transform",
      "config": {}
    }
  ],
  "schedule": "0 0 * * *"
}
```

**Response:** `201 Created`
```json
{
  "id": "pip-003",
  "name": "New ETL Pipeline",
  "description": "Pipeline description",
  "steps": [...],
  "status": "idle",
  "schedule": "0 0 * * *",
  "last_run": null,
  "created_at": "2024-01-15T11:00:00Z",
  "updated_at": "2024-01-15T11:00:00Z",
  "run_count": 0
}
```

### GET /api/pipelines/{pipeline_id}

Get a specific pipeline by ID.

**Path Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| pipeline_id | string | Pipeline ID |

**Response:** Same as single pipeline object.

### DELETE /api/pipelines/{pipeline_id}

Delete a pipeline.

**Response:**
```json
{
  "message": "Pipeline deleted successfully"
}
```

### POST /api/pipelines/{pipeline_id}/run

Trigger a pipeline run.

**Response:**
```json
{
  "id": "run-016",
  "pipeline_id": "pip-001",
  "pipeline_name": "HMP Data ETL Pipeline",
  "status": "running",
  "started_at": "2024-01-15T11:00:00Z",
  "finished_at": null,
  "duration_seconds": null,
  "steps_completed": 0,
  "total_steps": 4,
  "logs": ["[2024-01-15T11:00:00Z] Pipeline run started"],
  "error": null
}
```

### GET /api/pipelines/{pipeline_id}/runs

Get runs for a specific pipeline.

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| limit | integer | 20 | Maximum number of runs (max: 100) |

---

## Pipeline Runs

### GET /api/runs

List all pipeline runs.

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| limit | integer | 50 | Maximum number of runs (max: 200) |

### GET /api/runs/{run_id}

Get a specific run by ID.

### POST /api/runs/{run_id}/complete

Mark a run as complete.

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| success | boolean | true | Whether the run succeeded |
| error | string | null | Error message if failed |

---

## Experiments

### GET /api/experiments

List all experiments.

**Response:**
```json
[
  {
    "id": "exp-001",
    "name": "Activity Recognition Experiment 1",
    "description": "Training RandomForest model",
    "pipeline_id": "pip-002",
    "status": "completed",
    "parameters": {
      "algorithm": "RandomForest",
      "n_estimators": 100,
      "max_depth": 10,
      "learning_rate": 0.05
    },
    "metrics": {
      "accuracy": 0.8932,
      "precision": 0.8845,
      "recall": 0.8721,
      "f1_score": 0.8782,
      "training_time_seconds": 245.5
    },
    "model_version": "v1.0",
    "created_at": "2024-01-10T09:00:00Z",
    "finished_at": "2024-01-10T09:15:00Z"
  }
]
```

### POST /api/experiments

Create a new experiment.

**Request Body:**
```json
{
  "name": "New Classification Experiment",
  "description": "Testing GradientBoosting",
  "pipeline_id": "pip-002",
  "parameters": {
    "algorithm": "GradientBoosting",
    "n_estimators": 150,
    "max_depth": 15
  }
}
```

### GET /api/experiments/{experiment_id}

Get experiment by ID.

### PUT /api/experiments/{experiment_id}/metrics

Update experiment metrics.

**Request Body:**
```json
{
  "accuracy": 0.9123,
  "precision": 0.9045,
  "recall": 0.8987,
  "f1_score": 0.9015
}
```

### POST /api/experiments/{experiment_id}/complete

Mark experiment as complete and create model version.

**Request Body:**
```json
{
  "accuracy": 0.9123,
  "precision": 0.9045,
  "recall": 0.8987,
  "f1_score": 0.9015
}
```

**Response:**
```json
{
  "message": "Experiment completed",
  "model_version": "v2.0"
}
```

### DELETE /api/experiments/{experiment_id}

Delete an experiment.

---

## Models

### GET /api/models

List all registered models.

**Response:**
```json
[
  {
    "id": "model-001",
    "name": "activity_classifier",
    "version": "v1.0",
    "experiment_id": "exp-001",
    "algorithm": "RandomForest",
    "metrics": {
      "accuracy": 0.8932,
      "precision": 0.8845,
      "recall": 0.8721,
      "f1_score": 0.8782
    },
    "parameters": {
      "n_estimators": 100,
      "max_depth": 10
    },
    "file_path": "/models/activity_classifier/v1.0/model.pkl",
    "created_at": "2024-01-10T09:15:00Z",
    "status": "registered"
  }
]
```

### POST /api/models

Register a new model.

**Query Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| name | string | yes | Model name |
| experiment_id | string | yes | Associated experiment |
| algorithm | string | yes | Algorithm used |

**Request Body:**
```json
{
  "metrics": {"accuracy": 0.91},
  "parameters": {"n_estimators": 150}
}
```

### GET /api/models/{model_id}

Get model by ID.

---

## Data Validations

### GET /api/validations

List all data validations.

**Response:**
```json
[
  {
    "id": "val-001",
    "name": "Input Data Validation",
    "dataset_path": "/data/dataset_1.parquet",
    "status": "passed",
    "rules_passed": 12,
    "rules_failed": 0,
    "total_rules": 12,
    "issues": [],
    "profile": {
      "total_rows": 50000,
      "total_columns": 25,
      "missing_cells": 123,
      "duplicate_rows": 5,
      "memory_size_mb": 45.6
    },
    "created_at": "2024-01-15T08:00:00Z",
    "finished_at": "2024-01-15T08:01:00Z"
  }
]
```

### POST /api/validations

Create a new data validation.

**Request Body:**
```json
{
  "name": "New Validation",
  "dataset_path": "/data/new_dataset.parquet",
  "rules": [
    {"type": "not_null", "columns": ["id", "name"]},
    {"type": "unique", "columns": ["id"]},
    {"type": "range", "column": "age", "min": 0, "max": 120}
  ]
}
```

### GET /api/validations/{validation_id}

Get validation by ID.

---

## Logs

### GET /api/logs

List system logs with optional filtering.

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| limit | integer | 100 | Maximum logs (max: 500) |
| level | string | null | Filter by level (INFO, WARNING, ERROR, DEBUG) |
| source | string | null | Filter by source |

**Response:**
```json
[
  {
    "id": "log-001",
    "timestamp": "2024-01-15T10:30:00Z",
    "level": "INFO",
    "source": "Pipeline",
    "message": "Pipeline execution started",
    "metadata": {"run_id": "run-015"}
  }
]
```

### POST /api/logs

Create a new log entry.

**Query Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| level | string | yes | Log level |
| source | string | yes | Log source |
| message | string | yes | Log message |

**Request Body:**
```json
{
  "metadata": {"key": "value"}
}
```

---

## Seed Data

### POST /api/seed

Seed the database with sample data (for development/testing).

**Response:**
```json
{
  "message": "Database seeded successfully",
  "pipelines": 3,
  "runs": 15,
  "experiments": 10,
  "models": 8,
  "validations": 5,
  "logs": 50
}
```

---

## Error Handling

All errors return appropriate HTTP status codes with a JSON body:

```json
{
  "detail": "Error message describing what went wrong"
}
```

### Common Error Codes

| Status | Meaning |
|--------|---------|
| 400 | Invalid request format or parameters |
| 404 | Resource not found |
| 422 | Validation error (missing required fields) |
| 500 | Internal server error |
