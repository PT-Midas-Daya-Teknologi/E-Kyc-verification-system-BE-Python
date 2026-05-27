# Frame WebSocket server (Windows-friendly)

The full `requirments.txt` stack needs **dlib** (C++ build tools on Windows).  
For frontend frame transfer only, use the lightweight server:

## Quick start

```powershell
cd E-Kyc-verification-system-BE-Python
.\run-ws.ps1
```

Or manually:

```powershell
python -m venv .venv-ws
.\.venv-ws\Scripts\pip install -r requirements-ws.txt
.\.venv-ws\Scripts\python.exe -m uvicorn ws_server:app --host 0.0.0.0 --port 8000 --reload
```

- Health: http://localhost:8000/health
- WebSocket: ws://localhost:8000/ws
- Recordings: `frames/demo-*.avi`

## Full ML stack (optional)

Only if you need `main.py` face_recognition / DeepFace:

1. Install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) with **Desktop development with C++**
2. Install CMake
3. `pip install -r requirments.txt`
