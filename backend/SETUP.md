# Backend Setup Guide

This guide walks you through setting up, testing, and deploying the ETL-ML Backend API.

## Table of Contents

- [Local Development](#local-development)
- [Environment Variables](#environment-variables)
- [Docker Setup](#docker-setup)
- [Kubernetes Deployment](#kubernetes-deployment)
- [Railway Deployment](#railway-deployment)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)

---

## Local Development

### Prerequisites

- Python 3.11 or higher
- pip (Python package manager)
- MongoDB 6.0+ (locally or cloud instance)
- Git

### Installation Steps

#### 1. Clone the Repository

```bash
git clone https://github.com/Mattral/Integrated-ETL-and-Machine-Learning-Workflow-Management-System.git
cd backend
```

#### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

#### 3. Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install requirements
pip install -r requirements.txt
```

#### 4. Configure Environment Variables

Create a `.env` file in the backend directory:

```bash
# MongoDB Configuration
MONGO_URL=mongodb://localhost:27017
DB_NAME=etl_ml_dashboard

# AWS Configuration (Optional)
# AWS_ACCESS_KEY_ID=your_aws_key
# AWS_SECRET_ACCESS_KEY=your_aws_secret
# AWS_BUCKET_NAME=etl-ml-storage

# Application Settings
LOG_LEVEL=DEBUG
```

#### 5. Run the Backend

```bash
uvicorn server:app --host 0.0.0.0 --port 8001 --reload
```

The API will be available at `http://localhost:8001`

API Documentation: `http://localhost:8001/docs`

---

## Environment Variables

### Required Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `MONGO_URL` | MongoDB connection string | `mongodb://localhost:27017` |
| `DB_NAME` | Database name | `etl_ml_dashboard` |

### Optional Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `AWS_ACCESS_KEY_ID` | AWS access key for S3 | - |
| `AWS_SECRET_ACCESS_KEY` | AWS secret access key | - |
| `AWS_BUCKET_NAME` | S3 bucket name | `etl-ml-storage` |
| `LOG_LEVEL` | Application log level | `INFO` |

### Getting Environment Variables

#### MongoDB Connection String

**Local MongoDB:**
```
mongodb://localhost:27017
```

**MongoDB Atlas (Cloud):**
1. Go to [MongoDB Atlas](https://www.mongodb.com/cloud/atlas)
2. Create a project and cluster
3. Click "Connect" → "Connect your application"
4. Copy the connection string
5. Replace `<username>` and `<password>` with your credentials
6. Example: `mongodb+srv://user:password@cluster.mongodb.net/etl_ml_dashboard?retryWrites=true&w=majority`

#### AWS Credentials (Optional)

1. Go to [AWS Console](https://console.aws.amazon.com/)
2. Navigate to IAM → Users → Your User
3. Click "Create access key"
4. Copy Access Key ID and Secret Access Key
5. Add to `.env` file

#### Environment Variable Sources

**For Local Development:**
- Create `.env` file in backend directory
- Backend automatically loads from `.env`

**For Docker:**
- Pass via `-e` flag: `docker run -e MONGO_URL=... -e DB_NAME=...`
- Use `.env` file: `docker run --env-file .env`

**For Kubernetes:**
- Secrets: Update `k8s/configmap.yaml` and `k8s/secrets.yaml`
- ConfigMaps: Update configuration values

**For Railway:**
- Add in Railway project settings → Variables
- No need for `.env` file in production

---

## Docker Setup

### Building Docker Image

```bash
# Build the image
docker build -t etl-ml-backend:latest .

# Build with tag for registry
docker build -t your-registry/etl-ml-backend:latest .
```

### Running Docker Container

#### Local Development

```bash
# Run with local MongoDB
docker run -p 8001:8001 \
  -e MONGO_URL=mongodb://host.docker.internal:27017 \
  -e DB_NAME=etl_ml_dashboard \
  etl-ml-backend:latest
```

#### With Docker Compose (Recommended)

Create a `docker-compose.yml` in the backend directory:

```yaml
version: '3.9'

services:
  mongodb:
    image: mongo:latest
    ports:
      - "27017:27017"
    environment:
      MONGO_INITDB_ROOT_USERNAME: root
      MONGO_INITDB_ROOT_PASSWORD: password
    volumes:
      - mongodb_data:/data/db

  backend:
    build: .
    ports:
      - "8001:8001"
    environment:
      MONGO_URL: mongodb://root:password@mongodb:27017
      DB_NAME: etl_ml_dashboard
    depends_on:
      - mongodb

volumes:
  mongodb_data:
```

Run with:
```bash
docker-compose up -d
```

### Push to Container Registry

```bash
# Login to registry
docker login docker.io  # or your registry

# Tag image
docker tag etl-ml-backend:latest your-username/etl-ml-backend:latest

# Push image
docker push your-username/etl-ml-backend:latest
```

---

## Kubernetes Deployment

### Prerequisites

- Kubernetes cluster (GKE, EKS, AKS, or local minikube)
- `kubectl` CLI installed
- Helm (optional, for package management)

### Creating Namespace

```bash
kubectl create namespace etl-ml
```

### Deploying to Kubernetes

#### 1. Update ConfigMap and Secrets

Edit `k8s/configmap.yaml` with your MongoDB and AWS details:

```bash
kubectl apply -f k8s/configmap.yaml -n etl-ml
```

#### 2. Create Secrets

```bash
kubectl create secret generic etl-ml-secrets \
  --from-literal=mongo-url='mongodb://user:password@mongodb:27017' \
  --from-literal=aws-access-key-id='YOUR_KEY' \
  --from-literal=aws-secret-access-key='YOUR_SECRET' \
  -n etl-ml
```

#### 3. Deploy Application

```bash
# Update deployment.yaml with your registry
# Change: YOUR_REGISTRY/etl-ml-backend:latest

kubectl apply -f k8s/deployment.yaml -n etl-ml
kubectl apply -f k8s/service.yaml -n etl-ml
```

#### 4. Verify Deployment

```bash
# Check deployment status
kubectl get deployments -n etl-ml

# Check pods
kubectl get pods -n etl-ml

# Check services
kubectl get services -n etl-ml
```

#### 5. Access the Service

```bash
# Port forward for local access
kubectl port-forward service/etl-ml-backend 8001:8001 -n etl-ml

# Access at http://localhost:8001
```

### Scaling

```bash
# Scale to 3 replicas
kubectl scale deployment etl-ml-backend --replicas=3 -n etl-ml
```

### View Logs

```bash
# View logs from a pod
kubectl logs -f deployment/etl-ml-backend -n etl-ml

# Follow logs from all pods
kubectl logs -f deployment/etl-ml-backend -n etl-ml --all-containers=true
```

---

## Railway Deployment

Railway is a modern deployment platform perfect for this backend.

### Setup Steps

#### 1. Create Railway Account

- Go to [railway.app](https://railway.app)
- Sign up with GitHub
- Create a new project

#### 2. Connect Repository

```bash
# Initialize Railway in your project (optional)
railway init
```

#### 3. Configure Environment Variables

In Railway Dashboard:
1. Go to your project
2. Click "Variables"
3. Add these variables:

```
MONGO_URL=mongodb+srv://user:password@cluster.mongodb.net/etl_ml_dashboard
DB_NAME=etl_ml_dashboard
AWS_ACCESS_KEY_ID=your_key (optional)
AWS_SECRET_ACCESS_KEY=your_secret (optional)
AWS_BUCKET_NAME=etl-ml-storage (optional)
```

#### 4. Deploy

**Option A: From GitHub**
1. Connect your GitHub repository
2. Railway automatically deploys on push to main branch
3. Dockerfile is automatically detected

**Option B: Using Railway CLI**
```bash
# Login to Railway
railway login

# Deploy
railway up
```

#### 5. View Logs

```bash
railway logs
```

#### 6. Access Your Backend

Railway provides a public URL automatically:
- Example: `https://etl-ml-backend-production.up.railway.app`

You can also get the URL from Railway dashboard.

#### 7. Configure Domain (Optional)

1. In Railway Dashboard → Settings → Domain
2. Add custom domain or use Railway's auto-generated URL
3. Update frontend `REACT_APP_BACKEND_URL` accordingly

### Railway Best Practices

- ✅ Use environment variables for secrets (not hardcoded)
- ✅ Set proper resource limits
- ✅ Enable auto-deploy from main branch
- ✅ Monitor logs regularly
- ✅ Set up backup for MongoDB

---

## Testing

### Health Check

```bash
curl http://localhost:8001/api/health
```

### List All Pipelines

```bash
curl http://localhost:8001/api/pipelines
```

### Seed Database

```bash
curl -X POST http://localhost:8001/api/seed
```

### Run Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run tests
pytest tests/ -v
```

### Docker Health Check

```bash
# Check if container is healthy
docker ps --filter="status=running"

# Run command inside container
docker exec <container-id> curl http://localhost:8001/api/health
```

---

## Troubleshooting

### Cannot Connect to MongoDB

**Error**: `Connection refused`

**Solution**:
1. Verify MongoDB is running: `mongosh` or `mongo`
2. Check MongoDB URL in `.env`: Should match your setup
3. For cloud MongoDB, ensure IP whitelist is configured
4. Test connection: `python -c "from pymongo import MongoClient; MongoClient('your_url').test.command('ping')"`

### Docker Image Build Fails

**Error**: `pip install fails during build`

**Solution**:
1. Check internet connection
2. Try building with `--no-cache`: `docker build --no-cache -t etl-ml-backend:latest .`
3. Check requirements.txt for incompatible packages

### Port Already in Use

**Error**: `Address already in use`

**Solution**:
```bash
# Find process using port 8001
lsof -i :8001

# Kill process
kill -9 <PID>

# Or use different port
uvicorn server:app --port 8002
```

### Railway Deployment Fails

**Error**: `Build failed` or `Runtime error`

**Solution**:
1. Check Railway logs: `railway logs`
2. Verify Dockerfile is in backend directory
3. Ensure all required environment variables are set
4. Check that requirements.txt is valid

### Memory Issues in Kubernetes

**Error**: `Pod evicted` or `OOMKilled`

**Solution**:
1. Adjust resource limits in `k8s/deployment.yaml`
2. Increase limits: `resources.limits.memory: "1Gi"`
3. Check actual usage: `kubectl top pods -n etl-ml`

### WebSocket Connection Issues

**Error**: `WebSocket connection failed`

**Solution**:
1. Ensure backend is running
2. Check frontend URL configuration
3. Verify CORS settings in `server.py`
4. For Railway: ensure WebSocket is enabled in proxy settings

---

## Quick Reference

### Local Development Flow

```bash
# 1. Activate environment
source venv/bin/activate

# 2. Create .env file
cat > .env << EOF
MONGO_URL=mongodb://localhost:27017
DB_NAME=etl_ml_dashboard
EOF

# 3. Run backend
uvicorn server:app --reload

# 4. Open in browser
open http://localhost:8001/docs
```

### Docker Quick Commands

```bash
# Build
docker build -t etl-ml-backend:latest .

# Run
docker run -p 8001:8001 -e MONGO_URL=mongodb://... etl-ml-backend:latest

# Check logs
docker logs <container-id>

# Stop
docker stop <container-id>
```

### Kubernetes Quick Commands

```bash
# Deploy
kubectl apply -f k8s/ -n etl-ml

# Check status
kubectl get all -n etl-ml

# View logs
kubectl logs -f deployment/etl-ml-backend -n etl-ml

# Port forward
kubectl port-forward service/etl-ml-backend 8001:8001 -n etl-ml
```

---

## Support

For issues or questions:
1. Check the [main README](../README.md)
2. Review logs: `docker logs` or `kubectl logs`
3. Visit [Railway Docs](https://docs.railway.app)
4. Open GitHub issue: [Issues](https://github.com/Mattral/Integrated-ETL-and-Machine-Learning-Workflow-Management-System/issues)

---

**Last Updated**: February 2026
