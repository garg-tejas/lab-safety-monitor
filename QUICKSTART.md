# Quick Start - MarketWise Hackathon

## 🚀 Run the Application in 5 Minutes

### Prerequisites Check
```bash
# Verify installations
python --version    # Should be 3.9+
node --version      # Should be 16+
psql --version      # PostgreSQL
redis-cli --version # Redis
```

### Step 1: Start Services (Windows)

#### Option A: Using Docker (Easiest)
```bash
# Start PostgreSQL
docker run --name postgres-ppe -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=ppe_compliance -p 5432:5432 -d postgres:13

# Start Redis
docker run --name redis-ppe -p 6379:6379 -d redis:6
```

#### Option B: Windows Services
```bash
# Start PostgreSQL service (if installed)
# Start Redis (run redis-server.exe)
```

### Step 2: Setup Backend
```bash
cd backend

# Create virtual environment
python -m venv venv
venv\Scripts\Activate.ps1

# Install dependencies (takes 2-3 minutes)
pip install -r requirements.txt

# Configure environment
copy .env.example .env
# Edit .env if needed (default settings work with Docker setup above)

# Initialize database
python scripts/init_db.py

# Download models (takes 1-2 minutes)
python scripts/download_models.py
```

### Step 3: Setup Frontend
```bash
cd frontend

# Install dependencies (takes 2-3 minutes)
npm install
```

### Step 4: Launch Application

#### Terminal 1 - Backend
```bash
cd backend
venv\Scripts\Activate.ps1
python main.py
```
✅ Backend running at: **http://localhost:8000**

#### Terminal 2 - Frontend
```bash
cd frontend
npm start
```
✅ Frontend opens automatically at: **http://localhost:3000**

---

## 🎯 Test the System

### Test 1: Dashboard
1. Open http://localhost:3000
2. Should see dashboard with statistics (initially all zeros)

### Test 2: API Documentation
1. Open http://localhost:8000/docs
2. Browse available API endpoints
3. Try "GET /api/statistics"

### Test 3: Upload Video
1. Go to "Upload Video" page
2. Select any MP4 video file
3. Click "Upload Video"
4. Click "Process Video" after upload
5. Check "Detection Logs" for results

### Test 4: Live Webcam (Optional)
1. Go to "Live Feed" page
2. Click "Start Webcam"
3. Allow camera permissions
4. Should see live video feed

---

## 📊 Default Configuration

The system is pre-configured with:
- Database: `postgresql://postgres:postgres@localhost:5432/ppe_compliance`
- Redis: `localhost:6379`
- Backend: `http://localhost:8000`
- Frontend: `http://localhost:3000`

---

## 🔥 If Something Goes Wrong

### Backend won't start
```bash
# Check database connection
python -c "import psycopg2; psycopg2.connect('postgresql://postgres:postgres@localhost:5432/ppe_compliance')"

# Check Redis
redis-cli ping
```

### Frontend won't start
```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
npm start
```

### Models not downloading
```bash
# Manual download
cd backend
python
>>> from ultralytics import YOLO
>>> model = YOLO('yolov8n.pt')
>>> exit()
```

---

## ✅ You're Ready When:

- [x] Backend responds at http://localhost:8000/health
- [x] Frontend loads at http://localhost:3000
- [x] Dashboard displays (even with 0 stats)
- [x] API docs accessible at http://localhost:8000/docs
- [x] No errors in terminal logs

---

## 📝 Quick Command Reference

### Start Everything
```bash
# Terminal 1
cd backend && venv\Scripts\Activate.ps1 && python main.py

# Terminal 2
cd frontend && npm start
```

### Stop Everything
- Press `Ctrl+C` in both terminals
- Stop Docker containers: `docker stop postgres-ppe redis-ppe`

### Reset Database
```bash
cd backend
venv\Scripts\Activate.ps1
python scripts/init_db.py
```

---

**Total setup time: ~10 minutes (including downloads)**

**Good luck! 🎉**
