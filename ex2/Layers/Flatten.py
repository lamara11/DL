from Layers.Base import BaseLayer
import numpy as np
class Flatten(BaseLayer):
    def __init__(self):
      self.trainable=False

    def forward(self, input_tensor):
        self.input_shape = input_tensor.shape
        return input_tensor.reshape(input_tensor.shape[0], -1)

    def backward(self, error_tensor):
        return error_tensor.reshape(self.input_shape)
