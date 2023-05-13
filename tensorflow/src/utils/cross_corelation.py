import numpy as np


def correlate2d(in1: np.ndarray, in2: np.ndarray, mode="valid", stride=1):
    n = len(in1)
    m = len(in2)
    result = []
    result_shape = 0
    if (mode == "full"):
        result_shape = len(in1) + 1
        in1 = np.pad(in1, (m - 1, m - 1), mode='constant')
    elif (mode == "same"):
        result_shape = len(in1)
        in1 = np.pad(in1, (1, 1), mode='constant')
    else:
        result_shape = n - m + 1
    for row in range(result_shape):
        for col in range(result_shape):
            slider = in1[row:row+m, col:col+m,]
            result.append(np.sum(slider * in2))

    return np.array(result).reshape(result_shape, result_shape).astype(int)


def convolve2d(in1: np.ndarray, in2: np.ndarray, mode="valid", stride=1):
    return correlate2d(in1, np.rot90(in2, 2), mode, stride)
