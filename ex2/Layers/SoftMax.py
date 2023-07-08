import numpy as np
from Layers.Base import BaseLayer
class SoftMax(BaseLayer):
    """
        SoftMax activation for Neural-Network to scale logits/input into probabilites, implements a forward
        and backward pass taking input_tensor and error_tensor to return probabilities of each outcome
        """
    def __init__(self):
        super().__init__()
        self.input_tensor=None

    def forward(self, input_tensor):
        """
                Activation predcition (y_hat) for every element of batch:
                y_k = exp(input) / Σ exp(input)
                """
        input_tensor2=input_tensor-np.max(input_tensor)
        self.input_tensor=input_tensor
        exp_input = np.exp(input_tensor2)
        return exp_input / np.sum(exp_input, axis=1, keepdims=True)

    def backward(self, error_tensor):
        """
                error_n-1 = prediction * (error_n - Σ error_n * prediction)
                """
        predicted = self.forward(self.input_tensor)
        return predicted * (error_tensor - np.sum(error_tensor * predicted, axis=1, keepdims=True))
