import numpy as np
from activations import Activation
class Sigmoid(Activation):

    def callFunction(x):
        return 1 / (1 + np.exp(-x))
    def callDerivative(self, x):
        return self.callFunction(x) * (1- self.callFunction(x))