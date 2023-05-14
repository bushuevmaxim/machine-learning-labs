import sys
sys.path.insert(1, '/Users/max/projects/machine-learning-labs/tensorflow/src')
from loss_functions.cross_entropy import BinaryCrossEntropy
from loss_functions.mean_squared_error import MeanSquaredError
from activations.relu import Relu
from activations.tangh import TangH
from activations.sigmoid import Sigmoid
from neuro_network import Network
from layers.activation_layer import ActivationLayer
from layers.fully_connected_layer import FCLayer
from layers.сonvolutional_alyer import ConvolutionalLayer
from layers.flatten_layer import FlattenLayer
from layers.max_pooling_layer import MaxPoolingLayer

