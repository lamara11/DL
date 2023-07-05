import numpy as np
import copy
from Layers.SoftMax import SoftMax
from Layers.FullyConnected import FullyConnected
import pickle

class NeuralNetwork:
    """
    NeuralNetwork representing archtitecture of Neural-Network.
    """

    def __init__(self, optimizer,  weights_initializer, bias_initializer):
        # Initialize constructer variables
        self.regularizer = None  # Init
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
        regularization_loss = 0.0
        for layer in self.layers:
            #layer.testing_phase = False
            output = layer.forward(output)
            if self.optimizer.regularizer is not None and layer.trainable:
                regularization_loss += self.optimizer.regularizer.norm(layer.weights)
        res=0
        res += regularization_loss

        res = self.loss_layer.forward(output, self.label_tensor+res)
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

    @property
    def phase(self):
        return self.testing_phase

    @phase.setter
    def phase(self, value):
        self.testing_phase = value
        for layer in self.layers:
            layer.testing_phase = value


    def train(self, iterations):
        """
        train network and stores loss for each iteration
        """
        self.phase = True
        for iteration in range(iterations):
            output = self.forward()
            self.loss.append(output) #self.loss_layer.forward(output, self.label_tensor))
            self.backward(self.label_tensor)

    def test(self, input_tensor):
        """
        Propagates input through the network and returns predictionof the last layer
        """
        self.phase = False
        output = input_tensor
        for layer in self.layers:
            output = layer.forward(output)

        return output

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        del state['data_layer']
        return state

    def __setstate__(self, state):
        # Restore instance attributes
        self.__dict__.update(state)
        # Initialize the dropped members with None
        self.data_layer = None

def save(filename, net):
    with open(filename, 'wb') as f:
        pickle.dump(net, f)

def load(filename, data_layer):
    with open(filename, 'rb') as f:
        net = pickle.dump(f)
    net.data_layer = data_layer  # set data layer again
    return net
