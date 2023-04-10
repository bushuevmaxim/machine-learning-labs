import numpy as np
from activations import Activation
class Relu(Activation):
    def callFunction(self,x):
        if x > 0:
            return x
        return 0
    def callDerivative(self, x):
        if x >= 0:
            return 1
        return 0

