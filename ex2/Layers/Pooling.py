import numpy as np
from Layers.Base import BaseLayer
class Pooling(BaseLayer):
    def __init__(self, stride_shape, pooling_shape):
        super().__init__()
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape
        self.input_shape = None

    def forward(self, input_tensor):

        num_batches, num_channels, input_height, input_width = input_tensor.shape
        self.input_shape=input_tensor.shape
        stride_height = self.stride_shape[0]
        stride_width = self.stride_shape[1]
        pooling_height = self.pooling_shape[0]
        pooling_width = self.pooling_shape[1]

        output_height = int(np.ceil((input_height - pooling_height + 1) / stride_height))
        output_width = int(np.ceil((input_width - pooling_width + 1) / stride_width))

        output_tensor = np.zeros((num_batches, num_channels, output_height, output_width))
        self.max_x = np.zeros((*input_tensor.shape[0:2], output_height, output_width), dtype=int)
        self.max_y = np.zeros((*input_tensor.shape[0:2], output_height, output_width), dtype=int)

        p1 = 0
        for h in range(0, input_height - pooling_height + 1, stride_height):
            p2 = 0
            for w in range(0, input_width - pooling_width + 1, stride_width):
                temp = input_tensor[:, :, h:h + pooling_height, w:w + pooling_width].reshape(
                    *input_tensor.shape[0:2], -1)
                output_indices = np.argmax(temp, axis=2)

                x_indices = output_indices // pooling_width  # row indices
                y_indices = output_indices % pooling_width  # column indices

                self.max_x[:, :, p1, p2] = x_indices  # indices are assigned
                self.max_y[:, :, p1, p2] = y_indices

                output_tensor[:, :, p1, p2] = np.choose(output_indices, np.moveaxis(temp, 2, 0))
                p2 += 1

            p1 += 1

        return output_tensor



    """def backward(self, error_tensor):
        num_batches, num_channels, output_height, output_width = error_tensor.shape

        stride_height = self.stride_shape[0]
        stride_width = self.stride_shape[1]
        pooling_height = self.pooling_shape[0]
        pooling_width = self.pooling_shape[1]

        input_height = (output_height - 1) * stride_height + pooling_height
        input_width = (output_width - 1) * stride_width + pooling_width

        input_tensor = np.zeros((num_batches, num_channels, input_height, input_width))
        print("output shape")
        print(error_tensor.shape)
        for b in range(num_batches):
            for c in range(num_channels):
                for h in range(output_height):
                    for w in range(output_width):
                        h_start = h * stride_height
                        h_end = h_start + pooling_height
                        w_start = w * stride_width
                        w_end = w_start + pooling_width

                        max_index = self.max_indices[b, c, h, w]
                        max_h = max_index // pooling_width
                        max_w = max_index % pooling_width

                        input_tensor[b, c, h_start + max_h, w_start + max_w] = error_tensor[b, c, h, w]

        return input_tensor"""

    def backward(self, error_tensor):
        backward_output = np.zeros(self.input_shape)  # initialize with zeros using saved shape of input_tensor
        num_batches, num_channels, output_height, output_width = error_tensor.shape
        stride_height = self.stride_shape[0]
        stride_width = self.stride_shape[1]
        for a in range(num_batches):
            for b in range(num_channels):
                for i in range(output_height):
                    for j in range(output_width):
                        backward_output[a, b, i * stride_height + self.max_x[a, b, i, j],
                                                j * stride_width + self.max_y[a, b, i, j]] += error_tensor[a, b, i, j]

        return backward_output
