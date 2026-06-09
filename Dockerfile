FROM python:3.12

WORKDIR /code

COPY requirements.txt /code/requirements.txt

#RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get update && apt-get install tesseract-ocr tesseract-ocr-eng -y

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

ENV APP_ENV=docker

COPY ./app /code/app

COPY ./.env.docker /code

CMD ["fastapi", "run", "app/main.py", "--port", "80"]