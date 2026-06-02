import os
import uuid
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated

import io
import cv2
import numpy
import numpy as np
import base64
import json
import logging
import datetime
import pytesseract
from PIL import Image
from deepface import DeepFace
from sqlalchemy import create_engine, String, cast
from fastapi import FastAPI, WebSocket, UploadFile, Form, Query, WebSocketDisconnect
from sqlalchemy.orm import sessionmaker

from models.user_document import UserDocument

# Add Tesseract executable path 
pytesseract.pytesseract.tesseract_cmd = r'D:\tessract_ocr\tesseract.exe'

app = FastAPI()
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler('face_recognition.log')
logger.addHandler(file_handler)

# Align with Spring Boot datasource (application.yml uses password root)
engine = create_engine(
    os.getenv(
        "DATABASE_URL",
        "postgresql://postgres:root@localhost:5432/postgres",
    )
)

Session = sessionmaker(bind=engine)
session = Session()

# In-memory cache so the Java service can poll results while face match runs
_check_result_cache: dict[str, dict] = {}

# DeepFace face match is CPU-bound; never run it on the asyncio event loop
_face_match_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="face-match")

def cache_check_result(session_id, response: dict) -> None:
    _check_result_cache[str(session_id)] = response


def format_check_result(confidence: float, verified: bool) -> dict:
    face_score = f"{round(float(confidence), 1)}"
    final_result = "VERIFIED" if verified else "REJECTED"
    return {"face_score": face_score, "final_result": final_result}

def rescale(img):
    return cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)

def ocr_analysis(id_document_file):
    id_document_file_content = id_document_file.file.read()
    id_document_file_bytes = np.frombuffer(id_document_file_content, numpy.uint8)
    id_document = cv2.imdecode(id_document_file_bytes, cv2.IMREAD_UNCHANGED)
    id_document = cv2.cvtColor(id_document, cv2.COLOR_BGR2GRAY)
    ocr_data = pytesseract.image_to_data(rescale(id_document), output_type=pytesseract.Output.DICT)
    results = []
    n_boxes = len(ocr_data['text'])

    for i in range(n_boxes):
        if int(ocr_data['conf'][i]) > 90:
            word_data = {
                "text": ocr_data['text'][i],
                "left": ocr_data['left'][i],
                "top": ocr_data['top'][i],
                "width": ocr_data['width'][i],
                "height": ocr_data['height'][i],
                "confidence": ocr_data['conf'][i]
            }
            results.append(word_data)

    logger.info('OCR DATA %s', json.dumps(results))
    return results

def _compare_faces_sync(session_id: str, image_bytes: bytes, attempt_no: int) -> dict:
    """CPU-bound face match — must not run on the asyncio event loop."""
    db = Session()
    try:
        user_document_model = (
            db.query(UserDocument)
            .filter(cast(UserDocument.session_id, String) == cast(session_id, String))
            .first()
        )
        if user_document_model is None:
            logger.info({"error": "UserDocument not found for session_id", "session_id": session_id})
            formatted = format_check_result(0.0, False)
            response = {
                "session_id": session_id,
                "attempt_no": attempt_no,
                "confidence": 0.0,
                "verified": False,
                **formatted,
            }
            cache_check_result(session_id, response)
            return response

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_rgb = np.array(image)

        id_document_image = Image.open(io.BytesIO(user_document_model.content))
        id_document_image_rgb = np.array(id_document_image)

        result = DeepFace.verify(
            img1_path=image_rgb,
            img2_path=id_document_image_rgb,
            model_name="OpenFace",
            anti_spoofing=True,
        )
        logger.info("face_comparison_result: %s", result)

        confidence = float(result.__getitem__("confidence"))
        verified = bool(result.__getitem__("verified"))
        formatted = format_check_result(confidence, verified)
        response = {
            "session_id": session_id,
            "attempt_no": attempt_no,
            "confidence": confidence,
            "verified": verified,
            **formatted,
        }
        cache_check_result(session_id, response)
        return response
    except Exception as e:
        logger.error("exception %s", e)
        formatted = format_check_result(0.0, False)
        response = {
            "session_id": session_id,
            "attempt_no": attempt_no,
            "confidence": 0.0,
            "verified": False,
            **formatted,
        }
        cache_check_result(session_id, response)
        return response
    finally:
        db.close()


async def compare_faces(file, session_id, attempt_no):
    contents = await file.read()

    cache_check_result(
        session_id,
        {
            "session_id": str(session_id),
            "attempt_no": int(attempt_no),
            "face_score": "N/A",
            "final_result": "PROCESSING",
        },
    )

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        _face_match_executor,
        _compare_faces_sync,
        str(session_id),
        contents,
        int(attempt_no),
    )

def gen_frames():
    global camera
    global out
    try:
        camera = cv2.VideoCapture(0)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        file_name = "demo-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".avi"
        out = cv2.VideoWriter(file_name, fourcc, 10.0, (640, 480))
        timeout = int(3)
        is_success = False

        while camera.isOpened() and timeout > 0:
            success, frame = camera.read()
            if not success:
                logger.info({"error": "Error in getting frame from camera"})
                break
            else:
                ret, buffer = cv2.imencode('.jpg', frame)
                buffer_bytes = buffer.tobytes()

                # Frames-only (do not run DeepFace.analyze here; it blocks the server)
                # if cv2.waitKey(1) & 0xFF == ord('q'):  # quit when 'q' is pressed
                #     break
                timeout-=1
                yield (b' --frame \r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer_bytes + b'\r\n')

        if timeout < 0 and is_success is False:
            print({"error": "Face detection and comparison unsuccessful"})

    except Exception as e:
        logger.error(e)
        # print({"error": "General error"})
    finally:
        camera.release()
        out.release()
        cv2.destroyAllWindows()

# @app.route('/video_feed')
# def video_feed():
#     return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    file_name = "demo-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".avi"
    out = cv2.VideoWriter(file_name, fourcc, 10.0, (640, 480))

    try:
        while True:
            # Receive base64 encoded frame from React
            data = await websocket.receive_text()

            # Decode the base64 string to an image
            header, encoded = data.split(",", 1)
            data_bytes = base64.b64decode(encoded)
            np_data = np.frombuffer(data_bytes, dtype=np.uint8)
            frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)

            # Frames-only (do not run DeepFace.analyze here; it blocks /check_result)
            out.write(frame)
            # out.release()

            # (Optional) Send result back to React
            await websocket.send_text("Frame Processed")
    except WebSocketDisconnect:
        # Client closed the connection; nothing to do.
        pass
    except Exception as e:
        logger.exception("WebSocket error: %s", e)
    finally:
        out.release()

@app.post("/ocr_analysis")
async def do_ocr_analysis(id_document_file: UploadFile):
    return ocr_analysis(id_document_file)

@app.get("/check_result")
async def get_check_result(session_id: Annotated[str, Query()]):
    cached = _check_result_cache.get(str(session_id))
    if cached:
        return cached
    return {"face_score": "N/A", "final_result": "PENDING"}

@app.post("/check_result")
async def do_check_result(
    file: UploadFile,
    session_id: Annotated[str, Form()],
    attempt_no: Annotated[int, Form()] = 1,
):
    return await compare_faces(file, session_id, attempt_no)

if __name__ == '__main__':
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)