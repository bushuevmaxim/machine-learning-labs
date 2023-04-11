import numpy as np
from activations import Activation
class Relu(Activation):
    @staticmethod
    def callFunction(x):
        return np.maximum(0,x)
    @staticmethod
    def callDerivative( x):
        x[x<=0] = 0
        x[x>0] = 1
        return x

