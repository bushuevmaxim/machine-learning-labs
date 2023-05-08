from loss_functions.loss_function import *

class BinaryCrossEntropy(LossFunction):
    def fun(y_true, y_pred):
        return -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)).mean()
    
    def derivative(y_true,y_pred):
        
        return (-y_true/y_pred) + (1 - y_true)/(1 - y_pred)
    

def _softmax(X):
    exps = np.exp(X)
    return exps / np.sum(exps)