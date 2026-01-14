# 🚀 START HERE - Complete Setup Guide

## ✅ Implementation Status: COMPLETE

All code has been implemented! Now let's get it running.

---

## 📋 Step-by-Step Startup

### Option 1: Automated Setup (Recommended)

1. **Open PowerShell as Administrator** in the project root directory
2. Run the setup script:
   ```powershell
   .\start-services.ps1
   ```
3. Follow the on-screen instructions

### Option 2: Manual Setup

#### Step 1: Start Services (Choose One)

**Using Docker (Easiest):**
```powershell
# Start PostgreSQL
docker run --name postgres-ppe -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=ppe_compliance -p 5432:5432 -d postgres:13

# Start Redis
docker run --name redis-ppe -p 6379:6379 -d redis:6
```

**Or Install Locally:**
- PostgreSQL 13+ from postgresql.org
- Redis from redis.io or use WSL

#### Step 2: Setup Backend

**Open PowerShell Terminal 1:**
```powershell
cd backend

# Create virtual environment
python -m venv venv

# Activate it
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Copy environment file
copy .env.example .env

# Initialize database
python scripts\init_db.py

# Download models
python scripts\download_models.py

# Start backend server
python main.py
```

**Expected output:**
```
✓ Backend running at: http://localhost:8000
✓ API Docs at: http://localhost:8000/docs
```

#### Step 3: Setup Frontend

**Open PowerShell Terminal 2:**
```powershell
cd frontend

# Install dependencies
npm install

# Start development server
npm start
```

**Expected output:**
```
✓ Compiled successfully!
✓ Frontend running at: http://localhost:3000
```

---

## 🎯 Verification Checklist

After startup, verify:

- [ ] ✅ Backend responds: http://localhost:8000/health
- [ ] ✅ API docs load: http://localhost:8000/docs
- [ ] ✅ Frontend loads: http://localhost:3000
- [ ] ✅ Dashboard displays (even with 0 stats)
- [ ] ✅ No errors in terminal windows

---

## 🧪 Quick Test

1. **Open browser**: http://localhost:3000
2. **Go to "Upload Video"** page
3. **Select any MP4 video file**
4. **Click "Upload Video"**
5. **Click "Process Video"**
6. **Check "Detection Logs"** - you should see entries!

---

## 🐛 Troubleshooting

### Backend won't start

**Error: "ModuleNotFoundError"**
```powershell
cd backend
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Error: Database connection failed**
```powershell
# Check if PostgreSQL is running
docker ps | grep postgres-ppe

# If not running
docker start postgres-ppe

# Or recreate
docker rm postgres-ppe
docker run --name postgres-ppe -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=ppe_compliance -p 5432:5432 -d postgres:13
```

**Error: Redis connection failed**
```powershell
# Check if Redis is running
docker ps | grep redis-ppe

# If not running
docker start redis-ppe
```

### Frontend won't start

**Error: "Cannot find module"**
```powershell
cd frontend
rm -r node_modules
npm install
npm start
```

**Error: Port 3000 already in use**
```powershell
# Kill process on port 3000
netstat -ano | findstr :3000
taskkill /PID <PID> /F

# Or use different port
$env:PORT=3001
npm start
```

### Models not downloading

**Manual download:**
```powershell
cd backend
python
>>> from ultralytics import YOLO
>>> model = YOLO('yolov8n.pt')
>>> exit()
```

---

## 📊 What's Included

### Backend Features ✅
- FastAPI REST API
- PostgreSQL database
- YOLOv8 detection
- Face recognition
- DeepSORT tracking
- Compliance engine
- WebSocket support

### Frontend Features ✅
- React dashboard
- Video upload
- Live webcam feed
- Detection logs
- Statistics charts
- Real-time updates

---

## 🎬 Next Steps After Running

1. **Test all pages** - Dashboard, Live Feed, Upload Video, Logs
2. **Prepare demo video** - Record or download sample videos
3. **Train PPE model** (optional) - See SETUP_GUIDE.md
4. **Practice demo** - Prepare presentation
5. **Read NEXT_STEPS.md** - For hackathon preparation

---

## 📞 Quick Reference

**Backend:**
- URL: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

**Frontend:**
- URL: http://localhost:3000
- Dashboard: http://localhost:3000/
- Live Feed: http://localhost:3000/live
- Upload: http://localhost:3000/upload
- Logs: http://localhost:3000/logs

**Database:**
- Host: localhost
- Port: 5432
- Database: ppe_compliance
- User: postgres
- Password: postgres

**Redis:**
- Host: localhost
- Port: 6379

---

## ⚡ Quick Commands

**Start everything:**
```powershell
# Terminal 1
cd backend && .\start-backend.ps1

# Terminal 2
cd frontend && .\start-frontend.ps1
```

**Stop everything:**
```powershell
# Press Ctrl+C in both terminals
docker stop postgres-ppe redis-ppe
```

**Reset database:**
```powershell
cd backend
.\venv\Scripts\Activate.ps1
python scripts\init_db.py
```

---

## 🎉 You're Ready!

Your PPE Safety Compliance Detection System is fully implemented and ready to run!

**Total setup time:** ~10-15 minutes (including downloads)

**Need help?** Check:
- [QUICKSTART.md](QUICKSTART.md) - Fast setup
- [SETUP_GUIDE.md](SETUP_GUIDE.md) - Detailed guide
- [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md) - Technical details

**Good luck with your hackathon! 🚀**
