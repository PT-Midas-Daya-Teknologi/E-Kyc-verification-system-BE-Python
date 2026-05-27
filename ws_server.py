"""
Lightweight WebSocket frame server for E-KYC frontend integration.
No dlib / face_recognition / deepface required.
"""
import base64
import datetime
import logging
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ws_server")

app = FastAPI(title="E-KYC Frame WebSocket")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRAMES_DIR = Path(__file__).resolve().parent / "frames"
FRAMES_DIR.mkdir(exist_ok=True)


@app.get("/health")
def health():
    return {"status": "ok", "service": "frame-websocket"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_tag = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    file_name = FRAMES_DIR / f"demo-{session_tag}.avi"
    out = cv2.VideoWriter(str(file_name), fourcc, 10.0, (640, 480))
    frame_count = 0

    logger.info("WebSocket connected — recording to %s", file_name)

    try:
        while True:
            data = await websocket.receive_text()
            if "," not in data:
                await websocket.send_text("Invalid frame format")
                continue

            _, encoded = data.split(",", 1)
            data_bytes = base64.b64decode(encoded)
            np_data = np.frombuffer(data_bytes, dtype=np.uint8)
            frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)

            if frame is None:
                await websocket.send_text("Frame decode failed")
                continue

            if frame.shape[1] != 640 or frame.shape[0] != 480:
                frame = cv2.resize(frame, (640, 480))

            out.write(frame)
            frame_count += 1
            await websocket.send_text("Frame Processed")

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected (%s frames)", frame_count)
    except Exception as e:
        logger.error("WebSocket error: %s", e)
    finally:
        out.release()
        logger.info("Saved recording: %s", file_name)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("ws_server:app", host="0.0.0.0", port=8000, reload=True)
