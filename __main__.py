# from gpiozero import Button

import time 
import cv2 
import numpy as np
from camera.CameraController import  VideoCamera, ImageCamera
from UNet import UNet
from gpiozero import InputDevice
import threading

INPUT_PIN = 16
FPS = 50
WIDTH_0 = 2028
HEIGHT_0 = 1080
WIDTH_1 = 512
HEIGHT_1 = 512

landing_phase = [True]
input_pin = InputDevice(INPUT_PIN,True)
video = VideoCamera(0,'flight.mp4',FPS,(WIDTH_0,WIDTH_1))

Unet = UNet('models/model.tflite',5)

time.sleep(1)

video_thread = threading.Thread(target=video.take_video,args=(landing_phase,100))
video_thread.start()

while True:
    if input_pin.is_active:
        landing_phase[0] = False
        break

# landing system starts here

landing_phase[0] = False
time.sleep(1)

image_camera = ImageCamera(0)
image_camera.set_size(WIDTH_1,HEIGHT_1)
retval, image = video.capture_frame()

while True:
    retval, image = VideoCamera.capture_frame()
    if not retval:
        continue
    
    image_arr = np.array([image])
    prediction = Unet.predict(image_arr)







