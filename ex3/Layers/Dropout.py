import numpy as np

class Dropout:
    def __init__(self, keep_prob):
        self.keep_prob = keep_prob
        self.mask = None
        self.testing_phase = False
        self.trainable = False

    def forward(self, input_tensor):
        if not self.testing_phase:
            self.mask = np.random.binomial(1, self.keep_prob, size=input_tensor.shape) / self.keep_prob
            return input_tensor * self.mask
        return input_tensor

    def backward(self, error_tensor):
        if not self.testing_phase:
            return error_tensor * self.mask
        return error_tensor
