import numpy as np


def correlate2d(in1: np.ndarray, in2: np.ndarray, mode="valid", stride=1):
    y = []
    if (mode == "full"):
        in1 = np.pad(in1, 1, 'constant', constant_values=0)
    n = len(in1)
    m = len(in2)
    index1 = index3 = 0
    index2 = index4 = m
    shape = n - m + 1
    while (index2 <= n):
        while (index4 <= n):
            submatrix = in1[index1:index2, index3:index4]
            y.append(np.sum(np.multiply(submatrix, in2)))
            index3 += stride
            index4 += stride
        index1 += stride
        index2 += stride
        index3 = 0
        index4 = m

    return np.array(y).reshape((shape, shape))


def convolve2d(in1: np.ndarray, in2: np.ndarray, mode="valid", stride=1):
    correlate2d(in1, np.rot90(in2, 2), mode, stride)
