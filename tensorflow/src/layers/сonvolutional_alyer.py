import numpy as np
from layer import Layer
from scipy import signal


class ConvolutionalLayer(Layer):

    def __init__(self, input_shape, kernel_size, depth):
        input_height, input_width, input_depth = input_shape
        self.depth = depth
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_shape = (input_height - kernel_size +
                             1, input_width - kernel_size + 1, depth)
        self.kernels_shape = (kernel_size, kernel_size, input_depth)
        self.kernels = np.random.randn(*self.kernels_shape)
        self.biases = np.random.randn(*self.output_shape)

    def forward_propagation(self, input):
        self.input = input
        self.output = self.bias

        for i in range(self.depth):
            for j in range(self.input_depth):
                self.output = signal.correlate2d(
                    self.input[j], self.kernels[i, j], "valid")

        return self.output

    def backward_propagation(self, output_error, learning_rate):
        kernels_error = np.zeros(self.kernels_shape)
        input_error = np.zeros(self.input_shape)

        for i in range(self.depth):
            for j in range(self.input_depth):
                kernels_error[i, j] = signal.correlate2d(
                    self.input[j], output_error[i], "valid")
                input_error[j] += signal.convolve2d(
                    output_error[i], self.kernels[i, j], "full")

        self.kernels -= learning_rate * kernels_error
        self.biases -= learning_rate * output_error
        return input_error
