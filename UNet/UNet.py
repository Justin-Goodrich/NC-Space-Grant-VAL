import numpy as np
from tflite_runtime.interpreter import Interpreter

class UNet:
    def __init__(self,path,threads=None):
        self.interpreter = Interpreter(model_path=path,num_threads=threads)
        self.prepare_model()

    def prepare_model(self):
        self.interpreter.allocate_tensors()
        self.output = self.interpreter.get_output_details()[0]  
        self.input = self.interpreter.get_input_details()[0] 

    def predict(self, input_data):
        self.interpreter.set_tensor(self.input['index'], input_data)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output['index'])
        return output
    
