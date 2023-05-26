class Network:
    def __init__(self, verbose=True):
        self.layers = []
        self.loss = None
        self.loss_prime = None
        self.verbose = verbose

    def add(self, layer):
        self.layers.append(layer)

    def compile(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    def predict(self, input_data):
        output = input_data
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def fit(self, x_train, y_train, epochs, learning_rate):

        for e in range(epochs):
            error = 0
            for x, y in zip(x_train, y_train):
                output = self.predict(x)
                error += self.loss(y, output)
                grad = self.loss_prime(y, output)
                for layer in reversed(self.layers):
                    grad = layer.backward(grad, learning_rate)
            error /= len(x_train)
            if self.verbose:
                print(f"{e + 1}/{epochs}, error={error}")
