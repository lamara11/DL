import numpy as np
import copy
from Layers import Base
from Layers.FullyConnected import FullyConnected
from Layers.TanH import TanH


class RNN(Base.BaseLayer):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.trainable = True
        self._memorize = False

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.fcw = FullyConnected(hidden_size + input_size, hidden_size)
        self.weights = self.fcw.weights
        self.weights_w = None

        self.fcy = FullyConnected(hidden_size, output_size)
        # self.why = self.fcy.weights
        self.weights_y = None

        self.gradient_weights_n = np.zeros((self.hidden_size + self.input_size + 1, self.hidden_size))

        self.bptt = 0
        self.hidden_state = None
        self.prev_hidden_st = None
        self.time_steps = None
        self.optimizer = None

    def forward(self, input_tensor):
        self.time_steps = input_tensor.shape[0]

        self.hidden_state = np.zeros((self.time_steps + 1, self.hidden_size))

        if self._memorize:  # Not sure if its correct, but it only works this way
            if self.hidden_state is not None:
                self.hidden_state[0] = self.prev_hidden_st

        output_tensor = np.zeros((self.time_steps, self.output_size))  # init

        for t in range(self.time_steps):

            # match shape to concat
            xs = np.concatenate((np.expand_dims(self.hidden_state[t], 0), np.expand_dims(input_tensor[t], 0)), axis=1)

            self.hidden_state[t + 1] = TanH().forward(self.fcw.forward(xs))
            output_tensor[t] = (self.fcy.forward(self.hidden_state[t + 1][np.newaxis, :]))

        self.prev_hidden_st = self.hidden_state[-1]  # set it back to previous state
        self.input_tensor = input_tensor  # save for backward

        return output_tensor

    def backward(self, error_tensor):

        self.gradient_inputs = np.zeros((self.time_steps, self.input_size))

        self.gradient_weights_y = np.zeros((self.hidden_size + 1, self.output_size))
        self.gradient_weights_w = np.zeros((self.hidden_size + self.input_size + 1, self.hidden_size))

        count_bptt = 0

        grad_tanh = 1 - np.square(self.hidden_state[1::])  # compute hyperbolic tangent fnc # skip initial state

        hidden_error = np.zeros((1, self.hidden_size))

        for t in reversed(range(self.time_steps)):

            yh_error = self.fcy.backward(error_tensor[t][np.newaxis, :])
            self.fcy.input_tensor = np.hstack((self.hidden_state[t + 1], 1))[np.newaxis, :]

            grad_hidden = grad_tanh[t] * (hidden_error + yh_error)

            hidden_state_error = self.fcw.backward(grad_hidden)  # dhidden replaced
            hidden_error = hidden_state_error[:, 0:self.hidden_size]

            # assign grad
            self.gradient_inputs[t] = hidden_state_error[:, self.hidden_size:(self.hidden_size + self.input_size + 1)]

            # backpropagate through FCW
            # Update input of fcw for each timestep
            # self.fcw.input = xs
            # self._gradient_weights += self.fcw._gradient_weights

            self.fcw.input_tensor = np.hstack((self.hidden_state[t], self.input_tensor[t], 1))[np.newaxis, :]

            # backpropagation through time (BPTT) steps is less than or equal to the specified maximum BPTT steps
            """ It ensures that the gradients are only backpropagated over a limited number of previous time steps,
                as specified by the self.bptt parameter.
                This helps mitigate the vanishing or exploding gradient problem that can occur when backpropagating over long sequences."""
            if count_bptt <= self.bptt:
                self.weights_y = self.fcy.weights
                self.weights_w = self.fcw.weights
                self.gradient_weights_y = self.fcy.gradient_weights
                self.gradient_weights_w = self.fcw.gradient_weights

            count_bptt += 1  # counter increment

        if self.optimizer is not None:
            self.weights_y = self.optimizer.calculate_update(self.weights_y, self.gradient_weights_y)
            self.weights_w = self.optimizer.calculate_update(self.weights_w, self.gradient_weights_w)
            self.fcy.weights = self.weights_y
            self.fcw.weights = self.weights_w
        return self.gradient_inputs

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = copy.deepcopy(optimizer)

    @property
    def memorize(self):
        return self._memorize

    @memorize.setter
    def memorize(self, value):
        self._memorize = value

    def initialize(self, weights_initializer, bias_initializer):
        self.fcy.initialize(weights_initializer, bias_initializer)
        self.fcw.initialize(weights_initializer, bias_initializer)
        self.weights = self.fcw.weights  # to pass 1st test case

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, weights):
        self._weights = weights

    @property
    def gradient_weights(self):
        return self.gradient_weights_n

    @gradient_weights.setter
    def gradient_weights(self, gradient_weights):
        self.fcy.gradient_weights = gradient_weights