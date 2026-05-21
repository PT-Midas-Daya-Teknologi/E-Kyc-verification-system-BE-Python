import cv2
import json
import logging
import datetime
import pytesseract
import face_recognition
from flask_cors import CORS
from deepface import DeepFace
from flask import Flask, Response, request, jsonify

app = Flask(__name__)
CORS(app)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler('face_recognition.log')
logger.addHandler(file_handler)

def detect_facial_attribute_analysis(frame):
    demography = DeepFace.analyze(frame, actions=['age', 'gender', 'emotion', 'race'], enforce_detection=False,
                                  detector_backend='dlib')
    if demography is not None and len(demography) > 1:
        logger.debug('Detected %s faces', format(len(demography)))
        return None

    age = demography[0].get('age')
    gender = demography[0].get('gender')
    emotion = demography[0].get('dominant_emotion')
    race = demography[0].get('dominant_race')

    logger.debug("age %s, gender %s, emotion %s, race %s" ,str(age), gender, emotion, race)
    return True

def rescale(img):
    return cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)

def ocr_analysis():
    id_document = cv2.imread('Test_id_document.jpg')
    id_document = cv2.cvtColor(id_document, cv2.COLOR_BGR2GRAY)
    ocr_data = pytesseract.image_to_data(rescale(id_document), output_type=pytesseract.Output.DICT)
    results = []
    n_boxes = len(ocr_data['text'])

    for i in range(n_boxes):
        if int(ocr_data['conf'][i]) > 0.5:
            word_data = {
                "text": ocr_data['text'][i],
                "left": ocr_data['left'][i],
                "top": ocr_data['top'][i],
                "width": ocr_data['width'][i],
                "height": ocr_data['height'][i],
                "confidence": ocr_data['conf'][i]
            }
            results.append(word_data)

    logger.debug('OCR DATA %s', json.dumps(results))

def compare_faces(frame, image_path):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    image_from_frame_location = face_recognition.face_locations(rgb_frame, model='hog')
    logger.debug(f"image_from_frame_location: {image_from_frame_location}")

    if len(image_from_frame_location) == 0:
        logger.debug('image_from_frame_location is None')
        return None
    image_from_frame_encodings = face_recognition.face_encodings(rgb_frame, image_from_frame_location)
    logger.debug(f"image_from_frame_encodings: {image_from_frame_encodings}")
    if image_from_frame_encodings is not None and len(image_from_frame_encodings) > 1:
        logger.debug("Found more than {} face(s)".format(len(image_from_frame_encodings)))
        return None

    id_document_image = face_recognition.load_image_file(image_path)
    id_document_image_location = face_recognition.face_locations(id_document_image, model='hog')
    logger.debug(f"id_document_image_location: {id_document_image_location}")
    id_document_image_encodings = face_recognition.face_encodings(id_document_image, id_document_image_location)
    logger.debug(f"id_document_image_encodings: {id_document_image_encodings}")

    boolean_matches = face_recognition.compare_faces(image_from_frame_encodings, id_document_image_encodings[0])
    logger.debug(f"boolean_matches: {boolean_matches}")

    if True in boolean_matches:
        cv2.rectangle(
            frame,
            (image_from_frame_location[0][3], image_from_frame_location[0][0]),
            (image_from_frame_location[0][1], image_from_frame_location[0][2]),
            (0, 0, 255),
            2)

        cv2.rectangle(
            frame,
            (image_from_frame_location[0][3], image_from_frame_location[0][2] - 35),
            (image_from_frame_location[0][1], image_from_frame_location[0][2]),
            (0, 0, 255),
            cv2.FILLED
        )

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            frame,
            "TEST USER",
            (image_from_frame_location[0][3] + 6, image_from_frame_location[0][2] - 6),
            font,
            1.0,
            (255, 255, 255),
            1)
        return frame

    return None

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
                logger.debug({"error": "Error in getting frame from camera"})
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

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)