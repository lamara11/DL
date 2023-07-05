from Layers.Conv import Conv
from Layers.SoftMax import SoftMax
from Layers.ReLU import ReLU
from Layers.FullyConnected import FullyConnected
from NeuralNetwork import NeuralNetwork
from Optimization.Optimizers import Adam
from Optimization.Constraints  import L2Regularizer


def build():
    # Define architecture
    layers = [
        Conv((1, 1), (5, 5, 1), 6),
        ReLU(),
        # add more layers as needed...
        FullyConnected(10),
        SoftMax()
    ]

    # Create neural network
    net = NeuralNetwork(layers)

    # Define optimizer and regularizer
    optimizer = Adam(learning_rate=5e-4)
    regularizer = L2Regularizer(4e-4)
    optimizer.add_regularizer(regularizer)

    net.set_optimizer(optimizer)

    return net
