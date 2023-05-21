from loss_functions.loss_function import *


class BinaryCrossEntropy(LossFunction):
    def fun(y_true, y_pred):
        epsilon = 1e-8
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return np.mean(loss)

    def derivative(y_true, y_pred):
        epsilon = 1e-8
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return (y_pred - y_true) / ((1 - y_pred) * y_pred)


def _softmax(X):
    exps = np.exp(X)
    return exps / np.sum(exps)
