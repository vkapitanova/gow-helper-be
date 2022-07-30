FROM python:3.9

WORKDIR /python-docker
ADD requirements.txt /
RUN pip install -r /requirements.txt
# for opencv
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
# copy files
COPY . .

EXPOSE 5000
ENTRYPOINT ["./run.sh"]