class Network:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)
    
    def compile(self, loss_func, learning_rate):
        self.loss_function = loss_func
        self.learning_rate = learning_rate
    
    def fit(self, x_train, y_train, epochs):
        samples = len(x_train)
        for epoch in range(epochs):
            err = 0
            for j in range(samples):
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                error = self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, self.learning_rate)
            err /= samples

    def predict(self, input_data):
        samples = len(input_data)
        result = []

        for i in range(samples):
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)
        return result
        