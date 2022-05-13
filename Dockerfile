#
FROM python:3.9
#
WORKDIR /code
#
COPY ./requirements.txt /code/requirements.txt
#
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
#
COPY . .
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
EXPOSE 8000
#
CMD ["python", "main.py"]
