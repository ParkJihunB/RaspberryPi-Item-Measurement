import board
import busio
import adafruit_vl53l0x
#while True:
    #print('Range: {}mm'.format(sensor.range))
    #time.sleep(1)
    
class Laser:
    def __init__(self):
        i2c = busio.I2C(board.SCL, board.SDA)
        self.sensor = adafruit_vl53l0x.VL53L0X(i2c)

    def measure(self):
        return self.sensor.range/10