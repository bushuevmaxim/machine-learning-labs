import numpy as np
from layer import Layer
class FCLayer(Layer):
    def __init__(self, input_size, output_size, activation):
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5
        self.activation = activation
    
    def forward_propagation(self, input):
        self.input = input
        self.output = np.dot(input, self.weights) + self.bias
        return self.output

    def backward_propagation(self, output_error):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        self.weights -= self.learning_rate * weights_error
        self.bias -= self.learning_rate * output_error
        return input_error
