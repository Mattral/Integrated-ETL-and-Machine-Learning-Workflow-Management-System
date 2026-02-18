# Setup Guide

Complete setup instructions for the ETL & ML Platform.

## Prerequisites

### Required Software

| Software | Minimum Version | Recommended |
|----------|-----------------|-------------|
| Python | 3.8+ | 3.10+ |
| Node.js | 16+ | 18+ |
| MongoDB | 4.4+ | 6.0+ |
| Git | 2.0+ | Latest |

### Hardware Requirements

**Development:**
- CPU: 2 cores
- RAM: 4 GB
- Storage: 10 GB

**Production:**
- CPU: 4+ cores
- RAM: 8+ GB
- Storage: 50+ GB (depends on data volume)

---

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/your-org/etl-ml-platform.git
cd etl-ml-platform
```

### 2. Backend Setup

#### Create Python Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Linux/macOS)
source venv/bin/activate

# Activate (Windows)
.\venv\Scripts\activate
```

#### Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

#### Configure Environment

Create `/backend/.env`:

```env
MONGO_URL=mongodb://localhost:27017
DB_NAME=etl_ml_dashboard
```

### 3. Frontend Setup

#### Install Node Dependencies

```bash
cd frontend
yarn install
# or
npm install
```

#### Configure Environment

Create `/frontend/.env`:

```env
REACT_APP_BACKEND_URL=http://localhost:8001
```

### 4. Database Setup

#### Install MongoDB

**macOS (Homebrew):**
```bash
brew tap mongodb/brew
brew install mongodb-community@6.0
brew services start mongodb-community@6.0
```

**Ubuntu/Debian:**
```bash
wget -qO - https://www.mongodb.org/static/pgp/server-6.0.asc | sudo apt-key add -
echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu focal/mongodb-org/6.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-6.0.list
sudo apt-get update
sudo apt-get install -y mongodb-org
sudo systemctl start mongod
```

**Windows:**
Download and install from [MongoDB Download Center](https://www.mongodb.com/try/download/community)

**Docker:**
```bash
docker run -d -p 27017:27017 --name mongodb mongo:6.0
```

#### Verify MongoDB

```bash
mongosh
# Should connect successfully
```

---

## Running the Application

### Development Mode

**Terminal 1 - Backend:**
```bash
cd backend
source ../venv/bin/activate  # If using venv
uvicorn server:app --host 0.0.0.0 --port 8001 --reload
```

**Terminal 2 - Frontend:**
```bash
cd frontend
yarn start
```

### Access Points

| Service | URL |
|---------|-----|
| Dashboard | http://localhost:3000 |
| API | http://localhost:8001/api |
| API Docs | http://localhost:8001/api/docs |
| ReDoc | http://localhost:8001/api/redoc |

### Seed Sample Data

1. Open the dashboard at `http://localhost:3000`
2. Navigate to **Settings**
3. Click **Seed Database**

---

## Docker Setup (Alternative)

### Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  mongodb:
    image: mongo:6.0
    ports:
      - "27017:27017"
    volumes:
      - mongodb_data:/data/db

  backend:
    build: ./backend
    ports:
      - "8001:8001"
    environment:
      - MONGO_URL=mongodb://mongodb:27017
      - DB_NAME=etl_ml_dashboard
    depends_on:
      - mongodb

  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    environment:
      - REACT_APP_BACKEND_URL=http://localhost:8001
    depends_on:
      - backend

volumes:
  mongodb_data:
```

### Backend Dockerfile

Create `backend/Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8001"]
```

### Frontend Dockerfile

Create `frontend/Dockerfile`:

```dockerfile
FROM node:18-alpine

WORKDIR /app

COPY package.json yarn.lock ./
RUN yarn install

COPY . .

RUN yarn build

FROM nginx:alpine
COPY --from=0 /app/build /usr/share/nginx/html
EXPOSE 3000
CMD ["nginx", "-g", "daemon off;"]
```

### Run with Docker

```bash
docker-compose up -d
```

---

## Configuration Options

### Backend Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MONGO_URL` | MongoDB connection string | `mongodb://localhost:27017` |
| `DB_NAME` | Database name | `etl_ml_dashboard` |
| `LOG_LEVEL` | Logging level | `INFO` |

### Frontend Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `REACT_APP_BACKEND_URL` | Backend API URL | `http://localhost:8001` |

---

## Verification

### Check Backend

```bash
curl http://localhost:8001/api/health
```

Expected response:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:00:00Z",
  "version": "1.0.0",
  "database": "connected"
}
```

### Check Frontend

Open `http://localhost:3000` in browser - should see the dashboard.

### Check Database

```bash
mongosh
use etl_ml_dashboard
db.pipelines.countDocuments()
```

---

## Troubleshooting

### Backend Won't Start

**Issue:** `ModuleNotFoundError`
```bash
# Ensure virtual environment is activated
source venv/bin/activate
pip install -r requirements.txt
```

**Issue:** Port already in use
```bash
# Find process using port 8001
lsof -i :8001
# Kill it
kill -9 <PID>
```

### Frontend Won't Start

**Issue:** Dependencies not installed
```bash
cd frontend
rm -rf node_modules
yarn install
```

**Issue:** Port conflict
```bash
PORT=3001 yarn start
```

### Database Connection Failed

**Issue:** MongoDB not running
```bash
# Check status
sudo systemctl status mongod

# Start MongoDB
sudo systemctl start mongod
```

**Issue:** Wrong connection string
- Verify `MONGO_URL` in `.env`
- Check MongoDB is accessible: `mongosh`

### CORS Errors

If you see CORS errors in browser console:
1. Verify backend is running
2. Check `REACT_APP_BACKEND_URL` matches backend URL
3. Backend CORS is configured to allow frontend origin

---

## Next Steps

After successful setup:

1. **Seed Data**: Go to Settings → Seed Database
2. **Explore Dashboard**: View statistics and metrics
3. **Create Pipeline**: Add your first ETL pipeline
4. **Run Experiment**: Create an ML experiment
5. **Check Docs**: Review Architecture and API documentation

---

## Support

For issues and questions:
- Check [GitHub Issues](https://github.com/your-org/etl-ml-platform/issues)
- Review documentation in `/docs` folder
- Join our community Discord
