# Architecture Guide

## System Overview

The ETL & ML Platform is designed as a modern, scalable system for managing data pipelines and machine learning workflows. This document provides a comprehensive overview of the system architecture.

## High-Level Architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│                              CLIENT LAYER                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     React Dashboard (SPA)                            │   │
│  │  ┌──────────┐ ┌───────────┐ ┌────────────┐ ┌────────────────────┐   │   │
│  │  │Dashboard │ │ Pipelines │ │Experiments │ │  Data Validation   │   │   │
│  │  │  Page    │ │   Page    │ │    Page    │ │       Page         │   │   │
│  │  └──────────┘ └───────────┘ └────────────┘ └────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     │ HTTP/REST
                                     ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                              API LAYER                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     FastAPI Application                              │   │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐   │   │
│  │  │  Pipeline   │ │ Experiment  │ │    Model    │ │ Validation  │   │   │
│  │  │  Endpoints  │ │  Endpoints  │ │  Endpoints  │ │  Endpoints  │   │   │
│  │  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘   │   │
│  │  ┌─────────────────────────────────────────────────────────────┐   │   │
│  │  │              Middleware (CORS, Auth, Logging)               │   │   │
│  │  └─────────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     │ PyMongo
                                     ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                           DATA LAYER                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        MongoDB Database                              │   │
│  │  ┌───────────┐ ┌────────────┐ ┌──────────┐ ┌────────────────────┐   │   │
│  │  │ pipelines │ │ experiments│ │  models  │ │ data_validations   │   │   │
│  │  └───────────┘ └────────────┘ └──────────┘ └────────────────────┘   │   │
│  │  ┌───────────────────────┐ ┌──────────────────────────────────┐     │   │
│  │  │    pipeline_runs      │ │              logs                │     │   │
│  │  └───────────────────────┘ └──────────────────────────────────┘     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────────────┘
```

## Component Details

### Frontend (React Dashboard)

The frontend is built as a Single Page Application (SPA) using React with the following technologies:

- **React 18**: Core UI framework
- **Recharts**: Data visualization library
- **Tailwind CSS**: Utility-first styling
- **Lucide React**: Icon library

#### Key Components

| Component | Description |
|-----------|-------------|
| `DashboardPage` | Overview with stats, charts, and recent activity |
| `PipelinesPage` | Pipeline management and run history |
| `ExperimentsPage` | ML experiment tracking and model registry |
| `ValidationsPage` | Data quality monitoring |
| `LogsPage` | Centralized log viewer |
| `SettingsPage` | Configuration and administration |

### Backend (FastAPI)

The backend provides a RESTful API with the following features:

- **Async/Await**: Non-blocking I/O for better performance
- **Pydantic Models**: Type validation and serialization
- **OpenAPI/Swagger**: Auto-generated API documentation
- **CORS Support**: Cross-origin resource sharing

#### API Module Structure

```
server.py
├── Configuration & Database Setup
├── Pydantic Models (Request/Response)
├── Helper Functions
├── API Routes
│   ├── Health Check
│   ├── Dashboard Statistics
│   ├── Pipelines CRUD
│   ├── Pipeline Runs
│   ├── Experiments CRUD
│   ├── Models Registry
│   ├── Data Validations
│   └── Logs
└── Main Entry Point
```

### Database (MongoDB)

MongoDB is used for its flexibility with semi-structured data:

#### Collections Schema

**pipelines**
```json
{
  "id": "string",
  "name": "string",
  "description": "string",
  "steps": [
    {
      "id": "string",
      "name": "string",
      "type": "extract|transform|load|validate|train",
      "config": {},
      "position": {"x": 0, "y": 0}
    }
  ],
  "status": "idle|running|success|failed|pending",
  "schedule": "string (cron format)",
  "last_run": "ISO datetime",
  "created_at": "ISO datetime",
  "updated_at": "ISO datetime",
  "run_count": "integer"
}
```

**experiments**
```json
{
  "id": "string",
  "name": "string",
  "description": "string",
  "pipeline_id": "string",
  "status": "running|completed|failed",
  "parameters": {
    "algorithm": "string",
    "n_estimators": "integer",
    "max_depth": "integer"
  },
  "metrics": {
    "accuracy": "float",
    "precision": "float",
    "recall": "float",
    "f1_score": "float"
  },
  "model_version": "string",
  "created_at": "ISO datetime",
  "finished_at": "ISO datetime"
}
```

**models**
```json
{
  "id": "string",
  "name": "string",
  "version": "string (e.g., v1.0)",
  "experiment_id": "string",
  "algorithm": "string",
  "metrics": {},
  "parameters": {},
  "file_path": "string",
  "created_at": "ISO datetime",
  "status": "registered|deployed|archived"
}
```

## Data Flow

### Pipeline Execution Flow

```
1. User triggers pipeline run
         │
         ▼
2. API creates run record (status: running)
         │
         ▼
3. Pipeline status updated in database
         │
         ▼
4. Steps execute sequentially:
   ┌──────────────────────────────┐
   │ Extract → Transform → Load   │
   │      ↓         ↓         ↓   │
   │   Validate  Validate  Validate│
   └──────────────────────────────┘
         │
         ▼
5. Run completed (status: success/failed)
         │
         ▼
6. Logs and metrics recorded
```

### Experiment Tracking Flow

```
1. Create experiment with parameters
         │
         ▼
2. Execute training pipeline
         │
         ▼
3. Log metrics during training
         │
         ▼
4. Complete experiment
         │
         ▼
5. Register trained model
         │
         ▼
6. Model available in registry
```

## Security Considerations

### Current Implementation

- CORS enabled for frontend-backend communication
- Input validation via Pydantic models
- MongoDB connection string in environment variables

### Production Recommendations

1. **Authentication**: Add JWT-based authentication
2. **Authorization**: Role-based access control (RBAC)
3. **HTTPS**: Enable TLS for all communications
4. **Rate Limiting**: Prevent API abuse
5. **Secrets Management**: Use vault for credentials

## Scalability

### Horizontal Scaling

```
                    ┌──────────────┐
                    │ Load Balancer│
                    └──────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│ API Instance 1│ │ API Instance 2│ │ API Instance N│
└───────────────┘ └───────────────┘ └───────────────┘
        │                  │                  │
        └──────────────────┼──────────────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │  MongoDB Replica Set│
                └─────────────────────┘
```

### Performance Optimizations

1. **Database Indexing**: Create indexes on frequently queried fields
2. **Caching**: Add Redis for session and query caching
3. **Connection Pooling**: Configure MongoDB connection pool
4. **Async Operations**: Leverage FastAPI's async capabilities

## Monitoring & Observability

### Recommended Stack

- **Metrics**: Prometheus + Grafana
- **Logging**: ELK Stack (Elasticsearch, Logstash, Kibana)
- **Tracing**: Jaeger or Zipkin
- **Alerting**: PagerDuty or Opsgenie

### Key Metrics to Monitor

| Metric | Description |
|--------|-------------|
| Pipeline success rate | % of successful pipeline runs |
| API latency | Response time percentiles (p50, p95, p99) |
| Error rate | Number of 5xx errors per minute |
| Database connections | Active MongoDB connections |
| Model training time | Duration of experiment runs |

## Integration Points

### External Systems

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Data Sources  │────▶│  ETL Platform   │────▶│  Data Sinks     │
│  - S3/GCS       │     │                 │     │  - Databases    │
│  - Databases    │     │                 │     │  - Data Lakes   │
│  - APIs         │     │                 │     │  - ML Services  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                               │
                               ▼
                        ┌─────────────────┐
                        │ Orchestrators   │
                        │  - Airflow      │
                        │  - Kubernetes   │
                        └─────────────────┘
```

## Future Enhancements

1. **Real-time Streaming**: Apache Kafka integration
2. **Distributed Computing**: Spark cluster support
3. **Feature Store**: Centralized feature management
4. **A/B Testing**: Model comparison framework
5. **AutoML**: Automated hyperparameter tuning
