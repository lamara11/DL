import numpy as np
from Layers.Base import BaseLayer


class FullyConnected(BaseLayer):  # Inherits base layer
    """
    Represenets a FullyConnected NN layers, performs forward, backward pass and
    update parameter back to Neural Network
    """

    def __init__(self, input_size, output_size):
        # Initialize constructer variable
        super().__init__()
        self.trainable = True
        self.weights = np.random.uniform(0, 1, (input_size + 1, output_size))  # Random weights, +1 for bias column
        self.input_size = input_size
        self.output_size = output_size
        self._gradient_weights = None
        self._gradient_biases = None
        self._optimizer = None

    def forward(self, input_tensor):
        """
        output_tensor = weights * input_tensor + bias
        """
        self.input = input_tensor
        self.output = np.dot(np.concatenate((input_tensor, np.ones((input_tensor.shape[0], 1))),
                                            axis=1), self.weights) #+ self.biases
        return self.output

    def backward(self, error_tensor):
        """
        E_n-1 = weight.T * E_n
        weight_t+1 =  weight_t - mu * E_n * input_tensor.T
        (E_n = error for 'n' layer)
        """
        self._gradient_weights = np.dot(np.concatenate([self.input, np.ones((self.input.shape[0], 1))],
                                                       axis=1).T, error_tensor)

        self.gradient_input = np.dot(error_tensor, self.weights[:-1].T)

        if self._optimizer is not None:  # Check if optimizer is set for particular layer
            self.weights = self._optimizer.calculate_update(self.weights, self._gradient_weights)


        return self.gradient_input

    # Set getter and setter property for optimizer
    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value

    # Set getter and setter property for gradient parameters
    @property
    def gradient_weights(self):
        return self._gradient_weights

    @property
    def gradient_biases(self):
        return self._gradient_biases

    def initialize(self, weights_initializer, bias_initializer):
        self.weights = weights_initializer.initialize(self.weights.shape,self.input_size, self.output_size)

