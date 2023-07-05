import numpy as np
from Layers import Base

class TanH(Base.BaseLayer):
    def __init__(self):
        super().__init__() # use instead of initializing trainable; for previous codes as well

        self.activations = None

    def forward(self, input_tensor):

        self.activations = np.tanh(input_tensor)
        return self.activations

    def backward(self, error_tensor):

        return error_tensor * (1 - np.square(self.activations))