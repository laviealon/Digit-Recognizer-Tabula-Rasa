from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional
from data_manager import DataPair, collect_sample
from random import uniform
from math import exp


class NeuralNetwork:
    """A Neural Network with an input layer, 2 hidden layers,
    and an output layer.
    """

    def __init__(self, fl_size: int, ll_size: int, hl_size: int) -> None:
        self.fl_size = fl_size
        self.hl_size = hl_size
        self.ll_size = ll_size
        np.random.seed(1)
        # initialize list of weights matrices
        self.weights = []
        # matrix of weights between first layer and hidden layer 1
        self.weights.append(10 * np.random.random((hl_size, fl_size)) - 5)
        # matrix of weights between hidden layer 1 and hidden layer 2
        self.weights.append(10 * np.random.random((hl_size, hl_size)) - 5)
        # matrix of weights between hidden layer 2 and last layer
        self.weights.append(10 * np.random.random((ll_size, hl_size)) - 5)
        # initialize list of biases matrices
        self.biases = []
        # matrix of biases of hidden layer 1
        self.biases.append(10 * np.random.random((hl_size, 1)) - 5)
        # matrix of biases of hidden layer 2
        self.biases.append(10 * np.random.random((hl_size, 1)) - 5)
        # matrix of biases of last layer
        self.biases.append(10 * np.random.random((ll_size, 1)) - 5)

    def compute(self, first_layer_activations: np.array) -> np.array:
        # calculate activations of hidden layer 1 using sigmoid
        # of matrix multiplication
        z1 = np.dot(self.weights[0], first_layer_activations) + self.biases[0]
        l1 = sigmoid_vec(z1)
        # calculate activations of hidden layer 2 using sigmoid
        # of matrix multiplication
        z2 = np.dot(self.weights[1], l1) + self.biases[1]
        l2 = sigmoid_vec(z2)
        # calculate last layer activations (output) using
        # sigmoid of matrix multiplication
        z3 = np.dot(self.weights[2], l2) + self.biases[2]
        l3 = sigmoid_vec(z3)
        return l3

    def _get_a_derivs_array(self, l0: np.array, z1: np.array, z2: np.array,
                            z3: np.array, exp_outp: np.array) -> List[np.array]:
        """Compute the partial derivatives of the cost function wrt the
        activations of the neurons in this neural network."""
        arrays = []
        # compute z-layers (see README) as activations
        l1 = sigmoid_vec(z1)
        l2 = sigmoid_vec(z2)
        l3 = sigmoid_vec(z3)
        # initialize the list of the partial derivatives wrt to the first
        # layer's activations
        a0_lst = []
        for i in range(len(l0)):
            dCda = self._get_a_deriv(0, i, [l0, z1, z2, z3], exp_outp)
            a0_lst.append(dCda)
        a0 = np.array(a0_lst)
        arrays.append(a0)
        # initialize the list of the partial derivatives wrt to the second
        # layer's activations
        a1_lst = []
        for i in range(l1):
            dCda = self._get_a_deriv(1, i, [l0, z1, z2, z3], exp_outp)
            a1_lst.append(dCda)
        a1 = np.array(a1_lst)
        arrays.append(a1)
        # initialize the list of the partial derivatives wrt to the third
        # layer's activations
        a2_lst = []
        for i in range(l2):
            dCda = self._get_a_deriv(2, i, [l0, z1, z2, z3], exp_outp)
            a2_lst.append(dCda)
        a2 = np.array(a2_lst)
        arrays.append(a2)
        # initialize the list of the partial derivatives wrt to the final
        # layer's activations
        a3_lst = []
        for i in range(l3):
            dCda = self._get_a_deriv(3, i, [l0, z1, z2, z3], exp_outp)
            a3_lst.append(dCda)
        a3 = np.array(a3_lst)
        arrays.append(a3)
        # return final list of arrays
        return arrays

    def _get_a_deriv(self, layer_index: int, neuron_index: int,
                     layers: List[np.array], exp_outp: np.array) -> float:
        """Insert docstring.

        To be used for method <_get_a_derivs_array> *only*.

        Preconditions:
        - 0 <= <layer_index> <= 3
        - <neuron_index> is a valid index for the layer corresponding to
            <layer_index> (see README)
        - <layers> must be the list [l0, z1, z2, z3] from <_get_a_derivs_array>
        """
        if layer_index == 3:
            z3 = layers[layer_index]
            a = sigmoid_vec(z3)[neuron_index]
            y = exp_outp[neuron_index]
            a_deriv = 2 * (a - y)
            return a_deriv
        else:
            s = 0
            z_next_layer = layers[layer_index + 1]
            for k in range(len(z_next_layer)):
                w = self.weights[layer_index][k][neuron_index]
                sig = sigmoid_deriv(z_next_layer[k])
                dCda = self._get_a_deriv(layer_index + 1, k, layers, exp_outp)
                s += (w * sig * dCda)
            a_deriv = s
            return a_deriv


def sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(x))


def sigmoid_deriv(x: float) -> float:
    return sigmoid(x) * (1 - sigmoid(x))


def sigmoid_vec(x: np.array) -> np.array:
    sigmoid_v = np.vectorize(sigmoid)
    return sigmoid_v(x)


def sigmoid_deriv_vec(x: np.array) -> np.array:
    sigmoid_deriv_v = np.vectorize(sigmoid_deriv)
    return sigmoid_deriv_v(x)


if __name__ == '__main__':
    # Test to make sure Neural Network works.
    """ 
    network = NeuralNetwork(6, 2, 16, 10)
    network.init_first_layer([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    print(network.compute_result())
    """
    import doctest
    doctest.testmod()
