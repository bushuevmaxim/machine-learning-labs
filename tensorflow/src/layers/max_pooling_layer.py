import numpy as np
from layers.layer import Layer


class MaxPoolingLayer(Layer):
    def __init__(self, pool_size, stride):
        self.pool_size = pool_size
        self.stride = stride

    def forward_propagation(self, input_data):
        self.input = input_data
        m = self.pool_size[0]
        self.indexes = []
        result = []
        self.mask = np.zeros(input_data.shape)
        print(input_data.shape)
        print(input_data)
        for row in range(0, self.input.shape[0], self.stride):
            for col in range(0, self.input.shape[0], self.stride):
                slider = np.array(input_data[row:row+m, col:col+m,])
                max, ind = get_max(slider)
                self.indexes.append((ind[0] + row, ind[1] + col))
                result.append(max)
                self.mask[row+ind[0], col + ind[1]] = 1
        self.output = np.array(result).reshape(self.pool_size)
        return self.output

    def backward_propagation(self, output_error: np.ndarray, learning_rate):

        self.input_error = np.zeros(self.mask.shape)
        for value, index in zip(output_error.flatten(), self.indexes):
            self.input_error[index[0]][index[1]] = value
        return self.input_error


def get_max(slider: np.ndarray):
    max_value = -np.inf
    n = slider.shape[0]
    print(slider)
    i_max = j_max = 0
    for i in range(n):
        for j in range(n):
            if (slider[i][j] > max_value):
                max_value = slider[i][j]
                i_max = i
                j_max = j
    return max_value, (i_max, j_max)
