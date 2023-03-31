# from gpiozero import Button

import time 
import cv2 
import numpy as np
from camera.CameraController import  VideoCamera, ImageCamera
from UNet.UNet import UNet
from gpiozero import InputDevice
import threading

INPUT_PIN = 16
FPS = 25
WIDTH_0 = 2028
HEIGHT_0 = 1080
WIDTH_1 = 512
HEIGHT_1 = 512
FALLBACK_TIME = 60*90

VIDEO_FILENAME = './flight.avi'

t = time.time()
landing_phase = [True]
input_pin = InputDevice(INPUT_PIN,True)
video = VideoCamera(0,VIDEO_FILENAME,FPS,(WIDTH_0,HEIGHT_0))

Unet = UNet('models/model.tflite',5)

time.sleep(1)

video_thread = threading.Thread(target=video.take_video,args=(landing_phase,100))
video_thread.start()

while True:
    if input_pin.is_active or time.time()-t >= FALLBACK_TIME:
        landing_phase[0] = False
        break

# landing system starts here

landing_phase[0] = False
time.sleep(1)

image_camera = ImageCamera(0)
image_camera.set_size(WIDTH_1,HEIGHT_1)
retval, image = video.capture_frame()

i = 0
while True:
    retval, image = image_camera.capture_frame()
    if not retval:
        continue
    
    image_arr = np.array([image])
    prediction = Unet.predict(image_arr)
    image_camera.save_image('./image_{}.jpg'.format(i),image)
    i+=1








