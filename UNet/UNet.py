import numpy as np
import tensorflow as tf

class UNet:
    def __init__(self,path,threads=None):
        self.interpreter = tf.lite.Interpreter(model_content=path,num_threads=threads)
        self.prerpare_model()

    def prepare_model(self):
        self.interpreter.allocate_tensors()
        self.output = self.interpreter.get_output_details()[0]  
        self.input = self.interpreter.get_input_details()[0] 

    def predict(self, input_data):
        self.interpreter.set_tensor(self.input['index'], input_data)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output['index'])
        return output