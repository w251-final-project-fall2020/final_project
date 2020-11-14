import sys

import RPi.GPIO as GPIO
from hx711 import HX711
from scale import Scale

import I2C_LCD_driver
import paho.mqtt.client as mqtt

LOCAL_MQTT_HOST = "192.168.0.174"
LOCAL_MQTT_PORT = 1883
LOCAL_MQTT_TOPIC = "weight_detection"
THRESHOLD = 20

detect_flag = False

mylcd = I2C_LCD_driver.lcd()

# choose pins on rpi (BCM5 and BCM6)
hx = HX711(dout=5, pd_sck=6)

# HOW TO CALCULATE THE REFFERENCE UNIT
#########################################
# To set the reference unit to 1.
# Call get_weight before and after putting 1000g weight on your sensor.
# Divide difference with grams (1000g) and use it as refference unit.

scale = Scale(source=hx)

scale.setReferenceUnit(108)

scale.reset()
mylcd.lcd_display_string("TARING...", 1)
scale.tare()
mylcd.lcd_clear()
mylcd.lcd_display_string("READY", 1)

while True:

    time.sleep(0.5)

    try:
        val = scale.getMeasure()

        if detect_flag:
            if val < THRESHOLD:
                mylcd.lcd_clear()
                mylcd.lcd_display_string("READY", 1)
                detect_flag = False
        
        else:
            if val > THRESHOLD:

                formatted_val = "{0: 4.4f}".format(val)
                mylcd.lcd_clear()
                mylcd.lcd_display_string("WEIGHT DETECTED", 1)
                mylcd.lcd_display_string(formatted_val, 2)
                time.sleep(5)

                val = scale.getMeasure()

                try:
                    mqttclient = mqtt.Client()
                    mqttclient.connect(LOCAL_MQTT_HOST, LOCAL_MQTT_PORT, 60)
                    mqttclient.publish(LOCAL_MQTT_TOPIC, payload=val, qos=0, retain=False)
                except:
                    pass

                detect_flag = True
                
            
    except (KeyboardInterrupt, SystemExit):
        GPIO.cleanup()
        sys.exit()
