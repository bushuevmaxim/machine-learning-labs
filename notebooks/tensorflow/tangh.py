import numpy as np
from activations import Activation
class TangH(Activation):
    
    @staticmethod
    def callFunction(self, x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x)+np.exp(-x))
    
    @staticmethod
    def callDerivative(self, x):
        return 1 - self.callFunction(x) ** 2

