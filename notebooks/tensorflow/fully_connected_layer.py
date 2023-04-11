import numpy as np
from layer import Layer
from activations import Activation
class FCLayer(Layer):
    def __init__(self, input_size, output_size, activation, activation_der):
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5
        self.activation = activation
        self.activation_der = activation_der
    
    def forward_propagation(self, input):
        self.input = input
        self.output = np.dot(input, self.weights) + self.bias
        self.output =  self.activation(x=self.output )
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return self.activation_der(self.output) * input_error
