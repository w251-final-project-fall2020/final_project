import sys
import time

import RPi.GPIO as GPIO
from hx711 import HX711
from scale import Scale

import I2C_LCD_driver

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
scale.tare()

while True:
    time.sleep(0.5)
    try:
        val = scale.getMeasure()
        formatted_val = "{0: 4.4f}".format(val)
        mylcd.lcd_display_string(formatted_val, 1)
        print(formatted_val)
        
    except (KeyboardInterrupt, SystemExit):
        GPIO.cleanup()
        sys.exit()
