# 🛡️ PPE Safety Compliance Detection System - Setup Guide

## Quick Start Guide

This guide will help you set up and run the PPE Safety Compliance Detection System for the MarketWise Hackathon.

---

## 📋 Prerequisites

### Required Software
- **Python 3.9+** - [Download](https://www.python.org/downloads/)
- **Node.js 16+** - [Download](https://nodejs.org/)
- **PostgreSQL 13+** - [Download](https://www.postgresql.org/download/)
- **Redis 6+** - [Download](https://redis.io/download/)
- **Git** - [Download](https://git-scm.com/)

### Optional (Recommended)
- **CUDA-capable GPU** with CUDA Toolkit for faster processing
- **Conda/Miniconda** for Python environment management

---

## 🚀 Installation Steps

### 1. Clone or Navigate to Project Directory

```bash
cd "c:\Users\harsh\OneDrive - Indian Institute of Information Technology, Nagpur\IIIT Nagpur\6th Semester\Projects\MarketWise Hackathon"
```

### 2. Setup PostgreSQL Database

#### Windows:
1. Install PostgreSQL from the official website
2. Create a database:
```sql
CREATE DATABASE ppe_compliance;
CREATE USER ppe_user WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE ppe_compliance TO ppe_user;
```

#### Or use Docker:
```bash
docker run --name postgres-ppe -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=ppe_compliance -p 5432:5432 -d postgres:13
```

### 3. Setup Redis

#### Windows:
Download Redis from: https://github.com/microsoftarchive/redis/releases

#### Or use Docker:
```bash
docker run --name redis-ppe -p 6379:6379 -d redis:6
```

### 4. Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows PowerShell:
venv\Scripts\Activate.ps1
# Windows CMD:
venv\Scripts\activate.bat

# Install dependencies
pip install -r requirements.txt

# Create .env file
copy .env.example .env

# Edit .env file with your database credentials
# Update DATABASE_URL with your PostgreSQL connection string
```

#### Configure .env file:
```env
DATABASE_URL=postgresql://ppe_user:your_password@localhost:5432/ppe_compliance
REDIS_HOST=localhost
REDIS_PORT=6379
```

#### Download YOLO Models:
```bash
python scripts/download_models.py
```

#### Initialize Database:
```bash
python scripts/init_db.py
```

### 5. Frontend Setup

```bash
# Navigate to frontend directory (from project root)
cd frontend

# Install dependencies
npm install
```

---

## 🎯 Running the Application

### Start Backend Server

```bash
# In backend directory with venv activated
cd backend
venv\Scripts\Activate.ps1

# Start FastAPI server
python main.py

# Or with uvicorn directly
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Backend will be available at: **http://localhost:8000**
API Documentation: **http://localhost:8000/docs**

### Start Frontend Development Server

```bash
# In frontend directory (new terminal)
cd frontend

# Start React development server
npm start
```

Frontend will open automatically at: **http://localhost:3000**

---

## 📊 Using the Application

### 1. Dashboard
- View real-time compliance statistics
- See violation breakdown by PPE type
- Monitor compliance rate

### 2. Live Feed
- Start webcam for real-time detection
- View live PPE compliance monitoring
- See person tracking and face detection

### 3. Upload Video
- Upload pre-recorded videos for analysis
- Process videos for batch detection
- View processing status and results

### 4. Detection Logs
- Browse all detection events
- Filter by compliance status
- Export detection data

---

## 🔧 Model Training (Optional)

### Download PPE Dataset

1. **Roboflow Universe:**
   - Visit: https://universe.roboflow.com/search?q=ppe
   - Download in YOLOv8 format

2. **Kaggle Datasets:**
   - Hard Hat Workers Dataset
   - Construction Site Safety Image Dataset

### Train Custom PPE Model

```bash
# Organize dataset in YOLO format
# Update configs/ppe_dataset.yaml with your paths

# Train model
yolo train data=configs/ppe_dataset.yaml model=yolov8n.pt epochs=100 imgsz=640 batch=16

# Test model
yolo val model=runs/detect/train/weights/best.pt data=configs/ppe_dataset.yaml

# Copy trained model
copy runs\detect\train\weights\best.pt models\weights\yolov8_ppe.pt
```

---

## 🐛 Troubleshooting

### Database Connection Error
```bash
# Check PostgreSQL is running
# Verify DATABASE_URL in .env
# Ensure database 'ppe_compliance' exists
```

### Redis Connection Error
```bash
# Check Redis is running
redis-cli ping  # Should return PONG
```

### CUDA/GPU Issues
```bash
# Check CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# If False, system will use CPU (slower but functional)
```

### Model Download Issues
```bash
# Manually download YOLOv8 weights
# Visit: https://github.com/ultralytics/assets/releases
# Download yolov8n.pt and place in models/weights/
```

### Frontend Won't Start
```bash
# Clear npm cache
npm cache clean --force

# Delete node_modules and reinstall
rmdir /s /q node_modules
npm install
```

---

## 📁 Project Structure

```
MarketWise Hackathon/
├── backend/
│   ├── api/              # FastAPI routes
│   ├── database/         # Database models & connection
│   ├── services/         # Business logic (detection, tracking, etc.)
│   ├── schemas/          # Pydantic schemas
│   ├── scripts/          # Utility scripts
│   ├── models/weights/   # ML model weights
│   ├── config.py         # Configuration
│   └── main.py          # Application entry point
├── frontend/
│   ├── src/
│   │   ├── pages/       # React pages
│   │   ├── services/    # API client
│   │   └── App.js       # Main app component
│   └── public/
├── data/                # Videos, snapshots, uploads
└── README.md
```

---

## 🎥 Demo Video Recording Tips

1. **Prepare Sample Videos:**
   - Industrial setting with workers
   - Mix of compliant and non-compliant scenarios
   - Clear lighting and multiple persons

2. **Record Demo:**
   - Show dashboard with statistics
   - Upload and process a video
   - Demonstrate live webcam detection
   - Browse detection logs

3. **Highlight Features:**
   - Real-time person tracking
   - Face recognition
   - PPE detection accuracy
   - Multi-person handling
   - Compliance logging

---

## 📝 Presentation Tips

### Technical Implementation
- Explain YOLOv8 for detection
- Describe DeepSORT tracking
- Show face recognition pipeline
- Demonstrate database schema

### System Architecture
- Video processing flow
- Detection → Tracking → Compliance
- WebSocket real-time updates
- FastAPI + React architecture

### Innovation Points
- Spatial association logic for PPE
- Face embedding for person identification
- Temporal smoothing for accuracy
- Scalable microservices design

---

## 🚨 Important Notes

1. **Model Availability:** System works with base YOLO model but PPE detection is enhanced with custom training
2. **Performance:** GPU recommended for real-time detection (15-25 FPS). CPU works but slower (5-8 FPS)
3. **Privacy:** System stores face embeddings, not actual face images
4. **Demo Mode:** If models not fully trained, system uses mock detections for demonstration

---

## 📧 Support

For issues or questions during hackathon:
- Check API documentation: http://localhost:8000/docs
- Review logs in `backend/logs/app.log`
- Verify all services are running (PostgreSQL, Redis, Backend, Frontend)

---

## ✅ Pre-Hackathon Checklist

- [ ] All prerequisites installed
- [ ] PostgreSQL database created and running
- [ ] Redis server running
- [ ] Backend dependencies installed
- [ ] Frontend dependencies installed
- [ ] YOLOv8 base model downloaded
- [ ] Database initialized
- [ ] Backend server starts without errors
- [ ] Frontend server starts without errors
- [ ] Can access dashboard at localhost:3000
- [ ] API docs accessible at localhost:8000/docs
- [ ] Sample video ready for demo
- [ ] Webcam working for live detection

---

## 🎯 Success Criteria

Your system is ready when:
1. ✅ Dashboard shows statistics (even if 0 initially)
2. ✅ Can upload and process a video
3. ✅ Live webcam feed displays
4. ✅ Detection logs page loads
5. ✅ No critical errors in console/logs
6. ✅ Database stores compliance events

---

**Good luck with your hackathon! 🚀**
