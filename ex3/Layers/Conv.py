import numpy as np
import math
import copy
from scipy.signal import correlate, correlate2d, convolve2d
from Layers.Base import BaseLayer


class Conv(BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        self.trainable = True
        if type(stride_shape) == int:
            stride_shape = (stride_shape, stride_shape)
        elif len(stride_shape) == 1:
            stride_shape = (stride_shape[0], stride_shape[0])
        self.stride_shape = stride_shape
        self.conv2d = (len(convolution_shape) == 3)
        self.weights = np.random.uniform(size=(num_kernels, *convolution_shape))
        if self.conv2d:
            self.convolution_shape = convolution_shape
        else:
            self.convolution_shape = (*convolution_shape, 1)
            self.weights = self.weights[:, :, :, np.newaxis]
        self.num_kernels = num_kernels
        self.bias = np.random.uniform(size=(num_kernels,))
        self.gradient_weights = None
        self.gradient_bias = None
        self._optimizer = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        if input_tensor.ndim == 3:
            input_tensor = input_tensor[:, :, :, np.newaxis]

        #padding image
        padded_image = np.zeros((input_tensor.shape[0], input_tensor.shape[1],
                                 input_tensor.shape[2] + self.convolution_shape[1] - 1,
                                 input_tensor.shape[3] + self.convolution_shape[2] - 1))

        p1 = int(self.convolution_shape[1] // 2 == self.convolution_shape[1] / 2)
        p2 = int(self.convolution_shape[2] // 2 == self.convolution_shape[2] / 2)
        if self.convolution_shape[1] // 2 == 0 and self.convolution_shape[2] // 2 == 0:
            padded_image = input_tensor
        else:
            padded_image[:, :, (self.convolution_shape[1] // 2):-(self.convolution_shape[1] // 2) + p1,
            (self.convolution_shape[2] // 2):-(self.convolution_shape[2] // 2) + p2] = input_tensor

        input_tensor = padded_image

        # dimensions of the output
        h_cnn = np.ceil((padded_image.shape[2] - self.convolution_shape[1] + 1) / self.stride_shape[0])
        v_cnn = np.ceil((padded_image.shape[3] - self.convolution_shape[2] + 1) / self.stride_shape[1])

        output_tensor = np.zeros((input_tensor.shape[0], self.num_kernels, int(h_cnn), int(v_cnn)))
        self.output_shape = output_tensor.shape

        # loop through the number of examples
        for n in range(input_tensor.shape[0]):
            # loop through the number of filters
            for f in range(self.num_kernels):
                # loop through the height of the output
                for i in range(int(h_cnn)):
                    # loop through the width of the output
                    for j in range(int(v_cnn)):
                        # check if within weights limits
                        if ((i * self.stride_shape[0]) + self.convolution_shape[1] <= input_tensor.shape[2]) and (
                                (j * self.stride_shape[1]) + self.convolution_shape[2] <= input_tensor.shape[3]):
                            output_tensor[n, f, i, j] = np.sum(input_tensor[n, :,
                                                               i * self.stride_shape[0]:i * self.stride_shape[0] +
                                                                                        self.convolution_shape[1],
                                                               j * self.stride_shape[1]:j * self.stride_shape[1] +
                                                                                        self.convolution_shape[
                                                                                            2]] * self.weights[f, :, :,
                                                                                                  :])
                            output_tensor[n, f, i, j] += self.bias[f]
                        else:
                            output_tensor[n, f, i, j] = 0
        if not self.conv2d:
            output_tensor = output_tensor.squeeze(axis=3)  # just to solve error in 1d case
        return output_tensor

    def backward(self, error_tensor):

        self.error_tensor_reshaped = error_tensor.reshape(
            self.output_shape)  # reshaping to match shape of output_tensor

        # Can use //if not ... ==3//
        if not self.conv2d:  # check for 2D
            self.input_tensor = self.input_tensor[:, :, :, np.newaxis]

        # Initialize with zeros, to store upsampled error tensor
        self.upsampled_error_tensor = np.zeros(
            (self.input_tensor.shape[0], self.num_kernels, *self.input_tensor.shape[2:]))

        # Initialize return tensor to return gradient weights
        return_tensor = np.zeros(self.input_tensor.shape)

        # Initialize with zero, stores input tensor after padding
        self.padded_input_tensor = np.zeros(
            (*self.input_tensor.shape[:2], self.input_tensor.shape[2] + self.convolution_shape[1] - 1,
             self.input_tensor.shape[3] + self.convolution_shape[2] - 1))

        # Initialize gradient bias with zeros, will store the gradient with respect to the bias terms
        self.gradient_bias = np.zeros(self.num_kernels)

        # Initialize gradient weights with zeros, will store the gradient with respect to the weights
        self.gradient_weights = np.zeros(self.weights.shape)

        # padding along the height dimension (verticle) of the input tensor
        padding_verticle = int(np.floor(self.convolution_shape[2] / 2))

        # padding along the width dimension (horizontal) of the input tensor
        padding_horizontal = int(np.floor(self.convolution_shape[1] / 2))

        # Upsampling and filling unsampled error tensor
        for batch in range(self.upsampled_error_tensor.shape[0]):
            for kernel in range(self.upsampled_error_tensor.shape[1]):
                # gradient with respect to the bias
                self.gradient_bias[kernel] += np.sum(error_tensor[batch, kernel, :])

                for h in range(self.error_tensor_reshaped.shape[2]):
                    for w in range(self.error_tensor_reshaped.shape[3]):
                        self.upsampled_error_tensor[batch, kernel, h * self.stride_shape[0], w * self.stride_shape[1]] = \
                        self.error_tensor_reshaped[batch, kernel, h, w]

                for channel in range(self.input_tensor.shape[1]):
                    return_tensor[batch, channel, :] += convolve2d(self.upsampled_error_tensor[batch, kernel, :],
                                                                   self.weights[kernel, channel, :],
                                                                   'same')  # zero padding

            # Delete the padding
            for n in range(self.input_tensor.shape[1]):
                for h in range(self.padded_input_tensor.shape[2]):
                    for w in range(self.padded_input_tensor.shape[3]):
                        if (h > padding_horizontal - 1) and (h < self.input_tensor.shape[2] + padding_horizontal):
                            if (w > padding_verticle - 1) and (w < self.input_tensor.shape[3] + padding_verticle):
                                self.padded_input_tensor[batch, n, h, w] = self.input_tensor[
                                    batch, n, h - padding_horizontal, w - padding_verticle]

            for kernel in range(self.num_kernels):
                for c in range(self.input_tensor.shape[1]):
                    # convolution of the upsampled error tensor with the padded input tensor
                    self.gradient_weights[kernel, c, :] += correlate2d(self.padded_input_tensor[batch, c, :],
                                                                       self.upsampled_error_tensor[batch, kernel, :],
                                                                       'valid')  # valid padding

        if self._optimizer is not None:
            self.weights = self._optimizer.weights.calculate_update(self.weights, self.gradient_weights)
            self.bias = self._optimizer.bias.calculate_update(self.bias, self.gradient_bias)

        # In case of 2D input
        if not self.conv2d:
            return_tensor = return_tensor.squeeze(axis=3)

        return return_tensor

    def initialize(self, weights_initializer, bias_initializer):
        self.weights = weights_initializer.initialize(self.weights.shape, np.prod(self.convolution_shape),
                                                      np.prod(self.convolution_shape[1:]) * self.num_kernels)
        self.bias = bias_initializer.initialize(self.bias.shape, 1, self.num_kernels)

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
        self._optimizer.weights = copy.deepcopy(optimizer)
        self._optimizer.bias = copy.deepcopy(optimizer)


