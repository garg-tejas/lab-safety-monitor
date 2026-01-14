# Backend Startup Script
# Run this in a PowerShell terminal

Write-Host "Starting Backend Server..." -ForegroundColor Cyan

# Navigate to backend directory
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptPath

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
.\venv\Scripts\Activate.ps1

# Initialize database (first time only)
if (-not (Test-Path ".db_initialized")) {
    Write-Host "Initializing database..." -ForegroundColor Yellow
    python scripts\init_db.py
    if ($LASTEXITCODE -eq 0) {
        New-Item -Path ".db_initialized" -ItemType File | Out-Null
    }
}

# Download models (first time only)
if (-not (Test-Path "models\weights\yolov8n.pt")) {
    Write-Host "Downloading YOLO models..." -ForegroundColor Yellow
    python scripts\download_models.py
}

# Start server
Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "Starting FastAPI Server..." -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host "Backend URL: http://localhost:8000" -ForegroundColor Cyan
Write-Host "API Docs: http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop" -ForegroundColor Yellow
Write-Host ""

python main.py
