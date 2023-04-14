from loss_functions.loss_function import *

class BinaryCrossEntropy(LossFunction):
    def fun(X,y):
        m = y.shape[0]
        p = _softmax(X)
        log_likelihood = -np.log(p[range(m),y])
        loss = np.sum(log_likelihood) / m
        return loss
    
    def derivative(X,y):
        m = y.shape[0]
        grad = _softmax(X)
        grad[range(m),y] -= 1
        grad = grad/m
        return grad
    

def _softmax(X):
    exps = np.exp(X)
    return exps / np.sum(exps)