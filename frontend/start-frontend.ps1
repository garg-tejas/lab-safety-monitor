# Frontend Startup Script
# Run this in a PowerShell terminal

Write-Host "Starting Frontend Development Server..." -ForegroundColor Cyan

# Navigate to frontend directory
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptPath

# Check if node_modules exists
if (-not (Test-Path "node_modules")) {
    Write-Host "Installing dependencies..." -ForegroundColor Yellow
    npm install
}

# Start development server
Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "Starting React Development Server..." -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host "Frontend URL: http://localhost:3000" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop" -ForegroundColor Yellow
Write-Host ""

npm start
