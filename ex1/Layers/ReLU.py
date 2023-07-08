import numpy as np
from Layers.Base import BaseLayer

class ReLU(BaseLayer):
    """
        Rectified Linear Unit (RELU) for Neural-Network, implements a forward and backward pass
        taking input_tensor and error_tensor to return ReLU output (positive part of its argument)
        """
    def __init__(self):
        super().__init__()
        self.input_tensor=None

    def forward(self, input_tensor):
        """
        ReLU Activation function: f(x) = max(0, input)
        """
        self.input_tensor=input_tensor
        return np.maximum(0, input_tensor)

    def backward(self, error_tensor):
        """
               error_n-1 = 0; if input =< 0
                         = error_n; else
               """
        return np.where(self.input_tensor > 0, error_tensor, 0)