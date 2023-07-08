import numpy as np
import copy
from Layers.SoftMax import SoftMax
from Layers.FullyConnected import FullyConnected

class NeuralNetwork:
    """
    NeuralNetwork representing archtitecture of Neural-Network.
    """

    def __init__(self, optimizer,  weights_initializer, bias_initializer):
        # Initialize constructer variables
        self.optimizer = optimizer
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer

    def forward(self):
        """
        takes input from data layer and pass it through all layers in the neural network
        """
        input_tensor, self.label_tensor = self.data_layer.next()
        output = input_tensor
        for layer in self.layers:
            output = layer.forward(output)

        res = self.loss_layer.forward(output, self.label_tensor)
        return res

    def backward(self, label_tensor):
        """
        inputs labels and propagates it back through the network
        """
        error = self.loss_layer.backward(label_tensor)
        for layer in reversed(self.layers):  # back propagation
            error = layer.backward(error)

    def append_layer(self, layer):
        """
        stacks both trainable/non-trainable layers to the network
        """
        if layer.trainable:
            layer.initialize(self.weights_initializer,self.bias_initializer)
            layer.optimizer = copy.deepcopy(self.optimizer)  # makes a copy if True for optimizer

        self.layers.append(layer)

    def train(self, iterations):
        """
        train network and stores loss for each iteration
        """
        for iteration in range(iterations):
            output = self.forward()
            self.loss.append(output) #self.loss_layer.forward(output, self.label_tensor))
            self.backward(self.label_tensor)

    def test(self, input_tensor):
        """
        Propagates input through the network and returns predictionof the last layer
        """
        output = input_tensor
        for layer in self.layers:
            output = layer.forward(output)

        return output
