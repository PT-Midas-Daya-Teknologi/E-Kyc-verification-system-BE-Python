import uuid
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
from fastapi import FastAPI, WebSocket, UploadFile, Form
from sqlalchemy.orm import sessionmaker

from models.user_document import UserDocument

app = FastAPI()
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler('face_recognition.log')
logger.addHandler(file_handler)

engine = create_engine('postgresql://postgres:postgres@localhost:5432/postgres')

Session = sessionmaker(bind=engine)
session = Session()

def detect_facial_attribute_analysis(frame):
    logger.info('Inside detect_facial_attribute_analysis()')
    demography = DeepFace.analyze(frame, actions=['age', 'gender', 'emotion', 'race'], enforce_detection=False, detector_backend='dlib')
    if demography is not None and len(demography) > 1:
        logger.info('Detected %s faces', format(len(demography)))
        return None

    face_confidence = demography[0].get('face_confidence')
    age = demography[0].get('age')
    gender = demography[0].get('dominant_gender')
    emotion = demography[0].get('dominant_emotion')
    race = demography[0].get('dominant_race')

    logger.info("face_confidence %s age %s, gender %s, emotion %s, race %s" ,str(face_confidence), str(age), gender, emotion, race)

    logger.info('Exiting detect_facial_attribute_analysis()')
    return True

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

async def compare_faces(file, session_id, attempt_no):
    try:
        user_document_model = session.query(UserDocument).filter(cast(UserDocument.session_id, String) == cast(session_id, String)).first()
        session.close()
        if user_document_model is None:
            logger.info({"error": "UserDocument not found for session_id"})
            return {"session_id": session_id, "attempt_no": attempt_no, "confidence": 0.0, "verified": False}

        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        image_rgb = np.array(image)

        id_document_image = Image.open(io.BytesIO(user_document_model.content))
        id_document_image_rgb = np.array(id_document_image)

        result = DeepFace.verify(img1_path=image_rgb, img2_path=id_document_image_rgb, model_name="OpenFace", anti_spoofing=True)
        logger.info("face_comparison_result: %s", result)

        return {"session_id": session_id, "attempt_no": attempt_no, "confidence": result.__getitem__('confidence'), "verified": result.__getitem__('verified')}
    except Exception as e:
        logger.error("exception %s", e)
        return {"session_id": session_id, "attempt_no": attempt_no, "confidence": 0.0, "verified": False}

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

                new_frame_face_analysis = detect_facial_attribute_analysis(frame)

                if new_frame_face_analysis is not None:
                    ocr_analysis()
                    # face_recognition_frame = compare_faces(frame, 'Test_id_document.jpg')
                    # if face_recognition_frame is not None:
                    #     out.write(face_recognition_frame)
                    #     print({"success": "Face detection and comparison successful"})
                    #     is_success = True
                    #     # break
                    # else:
                    #     print({"error": "Error in face comparison"})
                else:
                    print({"error": "No face detected"})
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

            # --- PROCESS FRAME HERE (e.g., Face Detection) ---
            # Example: grayscale conversion
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detect_facial_attribute_analysis(frame)

            out.write(frame)
            # out.release()

            # (Optional) Send result back to React
            await websocket.send_text("Frame Processed")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await websocket.close()

@app.post("/ocr_analysis")
async def do_ocr_analysis(id_document_file: UploadFile):
    return ocr_analysis(id_document_file)

@app.post("/check_result")
async def do_check_result(file: UploadFile, session_id: Annotated[uuid.UUID, Form()], attempt_no: Annotated[int, Form()]):
    return await compare_faces(file, session_id, attempt_no)

if __name__ == '__main__':
    app.run(debug=True)