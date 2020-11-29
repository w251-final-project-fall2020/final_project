FROM ubuntu
RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app
COPY . /usr/src/app

RUN apt update && apt install -y libffi-dev python3-pip curl unzip
RUN pip3 install -U paho-mqtt boto3 pillow
RUN apt install -y python3-scipy python3-numpy