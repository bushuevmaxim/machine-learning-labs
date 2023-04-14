import numpy as np
from activations.activation import Activation
class Sigmoid(Activation):
    @staticmethod
    def callFunction(x):
        return 1 / (1 + np.exp(-x))
    @staticmethod
    def callDerivative(x):
        return Sigmoid.callFunction(x) * (1- Sigmoid.callFunction(x))
