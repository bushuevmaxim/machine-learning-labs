import numpy as np
from loss_functions.loss_function import LossFunction
class MeanSquaredError(LossFunction):
    def fun(y_true, y_pred):
        return np.mean(np.power(y_true-y_pred, 2))
    
    def derivative(y_true, y_pred):
        return 2*(y_pred-y_true)/y_true.size
