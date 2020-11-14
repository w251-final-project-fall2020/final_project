import time
import board
import neopixel

# On a Raspberry pi, use this instead, not all pins are supported
pixel_pin = board.D12

# The number of NeoPixels
num_pixels = 24

# The order of the pixel colors - RGB or GRB. Some NeoPixels have red and green reversed!
# For RGBW NeoPixels, simply change the ORDER to RGBW or GRBW.
ORDER = neopixel.RGBW

pixels = neopixel.NeoPixel(pixel_pin, num_pixels, pixel_order=ORDER)

pixels.fill((255, 255, 255, 255))
