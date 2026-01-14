# PPE Safety Compliance Detection System

AI-powered real-time PPE (Personal Protective Equipment) detection and compliance monitoring system for industrial and laboratory environments.

## 🚀 Features

- **Real-time Detection**: Live webcam feed with instant PPE compliance detection
- **Video Processing**: Upload and analyze video files for PPE violations
- **Face Recognition**: Track individual workers and their compliance history
- **Multi-PPE Support**: Detects helmets, safety shoes, goggles, and masks
- **Dashboard Analytics**: Real-time statistics and compliance metrics
- **WebSocket Updates**: Live notifications for compliance events
- **RESTful API**: Complete API for integration with existing systems

## 🛠️ Tech Stack

**Backend:**
- FastAPI (Python 3.11+)
- YOLOv8 (Ultralytics) for object detection
- FaceNet for face recognition
- DeepSORT for multi-object tracking
- PostgreSQL for data persistence
- Redis for caching
- WebSocket for real-time updates

**Frontend:**
- React 18
- TailwindCSS
- Recharts for analytics
- Socket.IO for real-time communication

## 📋 Prerequisites

- Docker & Docker Compose (recommended)
- OR:
  - Python 3.11+
  - Node.js 18+
  - PostgreSQL 13+
  - Redis 6+

## 🐳 Quick Start with Docker

1. **Clone the repository:**
```bash
git clone <your-repo-url>
cd MarketWise-Hackathon
```

2. **Start all services:**
```bash
docker-compose up -d
```

3. **Initialize the database:**
```bash
docker-compose exec backend python scripts/init_db.py
```

4. **Access the application:**
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

5. **Stop all services:**
```bash
docker-compose down
```

## 💻 Local Development Setup

### Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Create .env file
cp .env.example .env

# Initialize database
python scripts/init_db.py

# Download models
python scripts/download_models.py

# Start backend
python main.py
```

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start frontend
npm start
```

## 📁 Project Structure

```
.
├── backend/
│   ├── api/                 # API routes
│   ├── database/            # Database models and connection
│   ├── services/            # Business logic (detection, tracking, etc.)
│   ├── schemas/             # Pydantic schemas
│   ├── scripts/             # Utility scripts
│   ├── main.py              # Application entry point
│   ├── config.py            # Configuration
│   ├── requirements.txt     # Python dependencies
│   └── Dockerfile           # Backend Docker configuration
│
├── frontend/
│   ├── src/
│   │   ├── pages/           # React pages
│   │   ├── services/        # API client
│   │   └── App.js           # Main React component
│   ├── package.json         # Node dependencies
│   └── Dockerfile           # Frontend Docker configuration
│
├── docker-compose.yml       # Docker orchestration
├── README.md                # This file
└── Extras/                  # Additional documentation and scripts

```

## 🔧 Configuration

### Environment Variables (Backend)

Create `backend/.env`:

```env
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/ppe_compliance
REDIS_URL=redis://localhost:6379/0
SECRET_KEY=your-secret-key-here
ENVIRONMENT=development
CONFIDENCE_THRESHOLD=0.5
IOU_THRESHOLD=0.4
```

### Environment Variables (Frontend)

Create `frontend/.env`:

```env
REACT_APP_API_URL=http://localhost:8000
```

## 📊 API Documentation

Once the backend is running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Key Endpoints

- `POST /api/upload-video` - Upload video for processing
- `POST /api/process-webcam-frame` - Process single webcam frame
- `GET /api/detections` - Get all detection events
- `GET /api/persons` - List all tracked persons
- `GET /api/statistics` - Get compliance statistics
- `WebSocket /ws` - Real-time updates

## 🎯 Usage

### 1. Live Webcam Detection
- Navigate to "Live Feed"
- Click "Start Webcam"
- System will detect PPE in real-time

### 2. Video Upload
- Go to "Video Upload"
- Select a video file (MP4, AVI, MOV)
- Click "Process Video"
- View results in Detection Logs

### 3. View Analytics
- Dashboard shows real-time statistics
- Filter logs by person or time range
- Export data for reporting

## 🧪 Training Custom PPE Model

1. **Prepare dataset:**
   - Download from [Roboflow Universe](https://universe.roboflow.com/)
   - Or create custom dataset with annotations

2. **Train model:**
```bash
cd backend
source venv/bin/activate
yolo train data=configs/ppe_dataset.yaml model=yolov8n.pt epochs=100 imgsz=640
```

3. **Update configuration:**
   - Replace model path in `config.py`

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- YOLOv8 by Ultralytics
- FaceNet PyTorch implementation
- DeepSORT real-time tracking
- FastAPI framework
- React and TailwindCSS communities

## 📧 Contact

For questions or support, please open an issue in the repository.

## 🐛 Troubleshooting

### Docker Issues
```bash
# Rebuild containers
docker-compose build --no-cache

# View logs
docker-compose logs -f backend
docker-compose logs -f frontend
```

### Database Connection Issues
```bash
# Check if PostgreSQL is running
docker-compose ps postgres

# Reset database
docker-compose down -v
docker-compose up -d
```

### Model Loading Issues
- Ensure `yolov8n.pt` is downloaded
- Set `TORCH_LOAD_WEIGHTS_ONLY=False` environment variable

## 🚀 Deployment

For production deployment:

1. Update environment variables
2. Use production database credentials
3. Enable HTTPS
4. Configure CORS properly
5. Set up monitoring and logging
6. Use docker-compose in production mode

```bash
docker-compose -f docker-compose.prod.yml up -d
```

---

**Built for MarketWise Hackathon 2026**
