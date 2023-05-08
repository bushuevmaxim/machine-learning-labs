from layers.layer import Layer

class FlattenLayer(Layer):
    def __init__(self):
        pass
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = input_data.flatten().reshape((1,-1))
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        return output_error.reshape(self.input.shape)
    