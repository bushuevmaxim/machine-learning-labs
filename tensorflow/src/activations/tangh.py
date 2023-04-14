import numpy as np
from activations.activation import Activation
class TangH(Activation):
    
    @staticmethod
    def callFunction( x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x)+np.exp(-x))
    
    @staticmethod
    def callDerivative(x):
        return 1 - TangH.callFunction(x) ** 2

