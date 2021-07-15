from __future__ import annotations
import numpy as np
from typing import List, Tuple
from data_manager import DataPair


class NeuralNetwork:
    """A Neural Network with an input layer, 2 hidden layers,
    and an output layer.

    === Attributes ===
    fl_size: the size of the network's first layer (input layer).
    hl_size: the size of the network's 2 hidden layers.
    ll_size: the size of the network's last layer (output layer).
    weights: a list of 3 matrices containing the synaptic weights between
        the network's neurons.
    biases: a list of 3 matrices containing the biases the network's neurons
        (excludes the input layer neurons as they don't require biases).
    """
    fl_size: int
    hl_size: int
    ll_size: int
    weights: List[np.array]
    biases: List[np.array]

    def __init__(self, fl_size: int, ll_size: int, hl_size: int) -> None:
        """Initialize a neural network."""
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

    def compute(self, first_layer_activations: np.array) -> List[np.array]:
        """Compute the output of this neural network with
        <first_layer_activations> as input.

        Return a list of the following form (indices are listed in brackets):
        [output layer, (0)
        z1,            (1)
        z2,            (2)
        z3]            (3)

        Preconditions:
        - <first_layer_activations> must be an array of size
            <self.fl_size> by 1
        """
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

    def get_grad(self, data_pair: DataPair) -> \
            Tuple[List[np.array], List[np.array]]:
        """Return the gradient of the cost function of this neural network.

        The gradient is returned as tuple, where the first element is a list
        of arrays containing the partial derivative of the cost function wrt
        the network's weights, and the first element is a list of arrays
        containing the partial derivative of the cost function wrt the
        network's biases.

        Preconditions:
        - <exo_outp> is an array of size <self.ll_size> by 1.
        """
        inp = data_pair.inp
        exp_outp = data_pair.exp_output
        z_vals = self.compute(inp)[1:]
        z1 = z_vals[0]
        l1 = sigmoid_vec(z1)
        z2 = z_vals[1]
        l2 = sigmoid_vec(z2)
        z3 = z_vals[2]
        l3 = sigmoid_vec(z3)
        z_layers = [z1, z2, z3]
        layers = [inp, l1, l2, l3]
        a_derivs = self._get_a_derivs_array(inp, z1, z2, z3, exp_outp)
        # Compute the partial derivatives wrt the network's weights
        weights_derivs = []
        for L in range(1, 4):
            weight_derivs_L = []
            weights_array = self.weights[L - 1]
            for i in range(len(weights_array)):
                weight_derivs_L_i = []
                row = weights_array[i]
                for j in range(len(row)):
                    dzdw = layers[L][j]
                    dadz = sigmoid_deriv(z_layers[L - 1][i])
                    dCda = a_derivs[L][i]  # if modify method <_get_a_derivs_array, change to L - 1>
                    weight_derivs_L_i.append(dzdw * dadz * dCda)
                weight_derivs_L.append(weight_derivs_L_i[:])
            weights_derivs.append(np.array(weight_derivs_L[:]))
        # Compute the partial derivatives wrt the network's biases
        biases_derivs = []
        for L in range(1, 4):
            biases_derivs_L = []
            biases_array = self.biases[L - 1]
            for i in range(len(biases_array)):
                dzdb = -1
                dadz = sigmoid_deriv(z_layers[L - 1][i])
                dCda = a_derivs[L][i]
                biases_derivs_L.append(dzdb * dadz * dCda)
            biases_derivs.append(biases_derivs_L[:])
        return weights_derivs, biases_derivs

    def _get_a_derivs_array(self, l0: np.array, z1: np.array, z2: np.array,
                            z3: np.array, exp_outp: np.array) -> List[np.array]:
        """Compute the partial derivatives of the cost function wrt the
        activations of the neurons in this neural network.

        Return a list of 4 arrays, each array represeting a layer in the
        neural network. Each array contains the partial derivatives of the cost
        function wrt the activations of the neurons corresponding to the layer
        the array represents.

        Preconditions:
        - <l0>, <z1>, <z2>, and <z3> must be the appropriate size relative
            to the layer they correspond to.
        - <exp_outp> must be a numpy array of size <self.ll_size> by 1.
        """
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
