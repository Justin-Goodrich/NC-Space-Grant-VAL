import numpy as np
import os
import time
import tensorflow as tf
from UNet.UNet import UNet
from utils.Dataset import Dataset

class PowerDraw:
    def __init__(self, path,threads):
        self.UNet = UNet(path,threads)
        self.UNet.prepare_model()

        os.system('wget -P landcover.ai.v1 https://landcover.ai.linuxpolska.com/download/landcover.ai.v1.zip')
        os.system('unzip ./landcover.ai.v1/landcover.ai.v1.zip -d ./landcover.ai.v1')

        self.dataset = Dataset('test.txt')

    def continuous_prediction(self, time_limit):
        t = time.time()

        while time.time()-t < time_limit:
            input = self.dataset.get_image()
            x = self.UNet.predict(input)

    def continuous_video(self):
        pass

