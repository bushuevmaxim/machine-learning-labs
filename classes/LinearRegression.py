import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
class LinearRegression:
    def __init__(self, lr = 0.0001, iters = 5000):
        self.lr = lr
        self.iters = iters

    def transform_(self, x):
        return np.concatenate((np.ones((len(x), 1)), x), axis = 1)

    def loss_func(self, x, y, w):
        return sum((y - np.dot(x, w)) ** 2) / x.shape[0]

    def fit(self, x, y):
        dist = np.inf
        eps = 1e-4
        X = self.transform_(x)

        w = np.zeros(X.shape[1])
        iter = 0

        while dist > eps and iter <= self.iters:
            loss = self.loss_func(X, y, w)
            w = w - self.lr * 2 * np.dot(X.T, np.dot(X, w) - y) / X.shape[0]
            dist = np.abs(loss - self.loss_func(X, y, w))
            iter += 1

        self.w = w

    def predict(self, x):
        return np.dot(self.transform_(x), self.w)