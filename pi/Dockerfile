FROM balenalib/raspberry-pi-debian

RUN apt update && apt install -y python3-dev python3-rpi.gpio python3-smbus i2c-tools python3-pip curl unzip
RUN pip3 install -U paho-mqtt adafruit-circuitpython-neopixel