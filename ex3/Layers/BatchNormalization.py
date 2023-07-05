import numpy as np
import copy
from Layers import Helpers


class BatchNormalization:
    """
    Batch Normalization layer
    """

    def __init__(self, channels):
        """
        Constructor.

        Args:
            channels (int): Number of channels in the input tensor.
        """
        self.channels = channels
        self.trainable = True
        self.testing_phase = False
        self.moving_mean = None
        self._optimizer = None
        self.moving_variance = None
        self.moving_avg_decay = 0.8  # alpha

        # Initialize
        self.weights = np.ones(self.channels)  # gamma
        self.bias = np.zeros(self.channels)  # beta

    def forward(self, input_tensor):
        """
        Performs the forward pass for the Batch Normalization

        Args:
            input_tensor (array/tensor): The input tensor to the Batch Normalization layer.

        Returns:
            array/tensor: The output tensor after applying Batch Normalization during the training phase.
        """
        epsilon = 1e-15  # Set epi
        need_conv = False

        # if input_tensor.ndim != 3:
        if input_tensor.ndim == 4:  # Handle 3-D tensor by reformating input tensor
            need_conv = True
            input_tensor = self.reformat(input_tensor)  # def reformat below

        self.input_tensor = input_tensor
        self.input_size= input_tensor.shape

        if self.testing_phase:  # Enters testing phase
            self.mean = self.moving_mean
            self.variance = self.moving_variance
        else:  # Enters training phase
            self.mean = np.mean(input_tensor, axis=0)
            self.variance = np.var(input_tensor, axis=0)
            if self.moving_mean is None:
                self.moving_mean = copy.deepcopy(self.mean)
                self.moving_variance = copy.deepcopy(self.variance)
            else:
                # section 9
                self.moving_mean = self.moving_avg_decay * self.moving_mean + (1 - self.moving_avg_decay) * self.mean
                self.moving_variance = self.moving_avg_decay * self.moving_variance + (
                            1 - self.moving_avg_decay) * self.variance

        self.input_tensor_hat = (input_tensor - self.mean) / np.sqrt(self.variance + epsilon)  # section 8
        output_tensor = self.weights * self.input_tensor_hat + self.bias  # section 8

        if need_conv:  # handle output of 3-D tensor by reformating output tensor
            output_tensor = self.reformat(output_tensor)
        self.output_size=output_tensor.shape
        return output_tensor

    def reformat(self, input_tensor):
        """
        Args:
            input_tensor (tensor): input tensor to be reformatted

        Returns:
            tensor : reformated tensor for batch normalization
        """
        # section 13; can be (B, H, M, N) instead of (batch_size, h, w, no_channel)
        if input_tensor.ndim == 4:  # Check 3-D
            self.store_shape = input_tensor.shape
            batch_size, h, w, no_channel = input_tensor.shape  # (B, H, M, N)
            input_tensor = input_tensor.reshape(batch_size, h, w * no_channel)
            input_tensor = input_tensor.transpose(0, 2, 1)
            input_tensor = input_tensor.reshape(batch_size * w * no_channel, h)
            return input_tensor
        else:
            batch_size, h, w, no_channel = self.store_shape
            input_tensor = input_tensor.reshape(batch_size, w * no_channel, h)
            input_tensor = input_tensor.transpose(0, 2, 1)
            input_tensor = input_tensor.reshape(batch_size, h, w, no_channel)
            return input_tensor

    def backward(self, error_tensor):
        """_summary_

        Args:
            error_tensor (tensor): input error tensor

        Returns:
            tensor: gradiants wrt to weights
        """
        need_conv = False
        if error_tensor.ndim == 4:  # check if reformating is requried
            need_conv = True
            error_tensor = self.reformat(error_tensor)

        # Formula @ section 10
        delta_wrt_gamma = np.sum(error_tensor * self.input_tensor_hat, axis=0)  # Gradient with respect to weights
        delta_wrt_beta = np.sum(error_tensor, axis=0)  # bias

        # gradient with respect to the input
        gradient = Helpers.compute_bn_gradients(error_tensor, self.input_tensor, self.weights, self.mean, self.variance)

        if self._optimizer is not None:
            self.weights = self._optimizer.weight.calculate_update(self.weights, delta_wrt_gamma)
            self.bias = self._optimizer.bias.calculate_update(self.beta, delta_wrt_beta)

        if need_conv:  # check if reformatting output grad is required
            gradient = self.reformat(gradient)

        self.gradient_weights = delta_wrt_gamma
        self.gradient_bias = delta_wrt_beta

        return gradient

    # used gpt to set these
    @property
    def weights(self):
        return self.gamma

    @weights.setter
    def weights(self, gamma):
        self.gamma = gamma

    @property
    def bias(self):
        return self.beta

    @bias.setter
    def bias(self, beta):
        self.beta = beta

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
        self._optimizer.weight = copy.deepcopy(optimizer)
        self._optimizer.bias = copy.deepcopy(optimizer)

    def initialize(self, weights_initializer, bias_initializer):
        self.weights = weights_initializer.initialize(self.weights.shape,self.channels, self.channels)
        self.bias = bias_initializer.initialize(self.bias.shape,self.channels, self.channels)