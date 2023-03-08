import time 
import cv2
import os
import threading

class CameraController():
    def __init__(self, id=0):
        self.id = id
        self.camera = cv2.VideoCapture(id,cv2.CAP_V4L)

    def turn_on(self):
        self.camera.open()

    def capture_frame(self):
        retval, image = self.camera.read()
        return retval, image
    
    def set_size(self,width, height):
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

class ImageCamera(CameraController):
    def image_collection(self, increment, n_photos, dir):

        """ 
        Takes a collection of images containg n photos
        @param increment: time increment in seconds for to take photos in
        @param n_photos: number of photos to take
        @param dir: sub directory to save images to 
        """

        dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),dir)

        os.makedirs(dir_path, exist_ok=True)

        
        for i in range(n_photos):
            #files are to be save in a given directory, each file is to be named minimum_payload_image_(index).png            
            file_name = os.path.join(dir_path,'minimum_payload_image_{}.png'.format(i))
            image = self.capture_frame()

            # because of the limit computing power of the Raspberry Pi Zero a new thread is to start to write image to memory,
            # in order to save time and take advantage of our 15 second increment requirement
            
            t = threading.Thread(target = cv2.imwrite, args=(file_name,image))
            t.start()
            time.sleep(increment)
            t.join()


class VideoCamera(CameraController):
    def __init__(self,id=0,filename=None, frame_rate=50, size=(2028,1080)):
        super().__init__(id)
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.Writer = cv2.VideoWriter(filename,self.fourcc,frame_rate,size)
        self.set_size(size[0],size[1])
    
    def take_video(self,flag,time_limit):
        t = time.time()
        while time.time()-t < time_limit and flag[0]:
            retval, image = self.capture_frame()
            if not retval:
                break

            self.Writer.write(image)
        print('early release')
        self.Writer.release()
        self.camera.release()

