# Run lightweight frame WebSocket server (no dlib / face_recognition)
$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
$Venv = Join-Path $Root ".venv-ws"

$Py311 = "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe"
if (-not (Test-Path $Py311)) {
    $Py311 = "python"
}

if (-not (Test-Path $Venv)) {
    Write-Host "Creating virtual environment..."
    & $Py311 -m venv $Venv
}

$Python = Join-Path $Venv "Scripts\python.exe"

Write-Host "Installing dependencies (requirements-ws.txt)..."
& $Python -m pip install --upgrade pip
& $Python -m pip install -r (Join-Path $Root "requirements-ws.txt")

Write-Host "Starting WebSocket server on ws://localhost:8000/ws"
& $Python -m uvicorn ws_server:app --host 0.0.0.0 --port 8000 --reload
