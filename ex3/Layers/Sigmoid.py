import numpy as np
from Layers import Base

class Sigmoid(Base.BaseLayer):

    def __init__(self):
        super().__init__() # use instead of initializing trainable; for previous codes as well

        self.activations = None

    def forward(self, input_tensor):

        self.activations = 1 / (1 + np.exp(-input_tensor))
        return self.activations

    def backward(self, error_tensor):

        return error_tensor * self.activations * (1 - self.activations)