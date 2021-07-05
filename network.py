from __future__ import annotations
from typing import List, Tuple, Optional
from data_manager import TrainingDataPair, collect_sample
from random import uniform
from math import exp


class Neuron:
    """A node in a neural network.

    === Public Attributes ===
    activation: a node's activation. If set to None, this Neuron is 'turned
        off', i.e. it does not have an activation assigned to it.
    next: a list of tuples, where the first entry is a neuron in the next layer
        that this neuron connects to, and the second entry is the weight of
        the connection.
    prev: a list of tuples, where the first entry is a neuron in the previous
        layer that this neuron connects to before, and the second entry is
        the weight of the connection.
    bias: the bias of the previous layer neurons' connections to this neuron.
        This is set to None when the neuron is part of a network's first layer
        (as it has no connections from the previous layer) or if simply
        a bias has not yet been assigned.

    === Representation Invariants ===
    - a neuron's activation must be between 0 and 1.
    """
    activation: Optional[float]
    next: List[Tuple[Neuron, float]]
    prev: List[Tuple[Neuron, float]]
    bias: Optional[float]

    def __init__(self, bias: float) -> None:
        """Insert docstring."""
        self.activation = None
        self.next = []
        self.prev = []
        self.bias = bias

    def add_nextlayer_connection(self, next_neuron: Neuron, weight: float)\
            -> None:
        """Insert docstring."""
        tup = next_neuron, weight
        self.next.append(tup)
        next_neuron.prev.append((self, weight))

    def change_connection_weight(self, prev_index: int, new_weight: float)\
            -> None:
        """Insert docstring.

        Preconditions:
            - <self.prev> (and thereby <self.next>) has at least
            <prev_index + 1> elements.
        """
        self.prev[prev_index] = self.prev[prev_index][0], new_weight
        self.next[prev_index] = self.next[prev_index][0], new_weight

    def get_z(self) -> float:
        """Get the z-value of this neuron."""
        s = 0
        for tup in self.prev:
            s += tup[0].activation * tup[1]
        s -= self.bias
        return s

    def activate(self) -> None:
        """Activate this neuron based on the values in <self.prev>.

        Preconditions:
            - all neurons in <self.prev> have either been activated or
            are the first layer of a neural network which has been initialized.
        """
        z = self.get_z()
        self.activation = sigmoid(z)


class NeuralNetwork:
    """ A Neural Network.

    === Public Attributes ===
    first_layer: a list of the neurons in this network's first layer.
    _middle_layers: a list of lists of the neurons in this network's
        middle layers.
    _last_layer: a list of the neurons in this network's last layer.

    === Representation Invariants ===
    - there must be middle layers, i.e. self._middle_layers cannot be an
        empty set.
    """
    first_layer: List[Neuron]
    _middle_layers: List[List[Neuron]]
    _last_layer: List[Neuron]

    def __init__(self, sizeof_first_layer: int, num_middle_layers: int,
                 sizeof_middle_layers: int, sizeof_last_layer) -> None:
        """ Insert docstring.


        """
        # Initialize the first layer of the network, each neuron with
        # a random bias
        first_layer = []
        for _ in range(sizeof_first_layer):
            rand_bias = uniform(-1, 1) * 10
            first_layer.append(Neuron(rand_bias))
        self.first_layer = first_layer
        # Initialize the middle layers of the network, each neuron with
        # a random bias
        middle_layers = []
        for _ in range(num_middle_layers):
            curr_middle_layer = []
            for _ in range(sizeof_middle_layers):
                rand_bias = uniform(-1, 1) * 10
                curr_middle_layer.append(Neuron(rand_bias))
            middle_layers.append(curr_middle_layer)
        self._middle_layers = middle_layers
        # Initialize the last layer of the network, each neuron with
        # a random bias
        last_layer = []
        for _ in range(sizeof_last_layer):
            rand_bias = uniform(-1, 1) * 10
            last_layer.append(Neuron(rand_bias))
        self._last_layer = last_layer
        # Assign random weights
        # first layer
        for neuron1 in self.first_layer:
            for neuron2 in self._middle_layers[0]:
                rand_weight = uniform(-1, 1) * 10
                neuron1.add_nextlayer_connection(neuron2, rand_weight)
        # middle layers, except last middle layer
        for layer_index in range(len(self._middle_layers[:-1])):
            curr_layer = self._middle_layers[layer_index]
            next_layer = self._middle_layers[layer_index + 1]
            for neuron1 in curr_layer:
                for neuron2 in next_layer:
                    rand_weight = uniform(-1, 1) * 10
                    neuron1.add_nextlayer_connection(neuron2, rand_weight)
        # last layer
        for neuron1 in self._middle_layers[-1]:
            for neuron2 in self._last_layer:
                rand_weight = uniform(-1, 1) * 10
                neuron1.add_nextlayer_connection(neuron2, rand_weight)

    def _set_first_layer(self, activations: List[float]) -> None:
        """Insert docstring.

        To be called by the method NeuralNetwork._compute_result *only*.

        Preconditions:
            - len(activations) is equal to len(self.first_layer).
            - all floats in <activations> are between 0 and 1.
        """
        for i in range(len(self.first_layer)):
            activation = activations[i]
            neuron = self.first_layer[i]
            neuron.activation = activation

    def _compute_result(self, activations: List[float]) -> List[float]:
        """Set the first layer of this network and compute its output.

        """
        # set first layer
        self._set_first_layer(activations)
        # calculate the activations of all neurons in middle layers
        for layer in self._middle_layers:
            for neuron in layer:
                s = 0  # initialize sum of weighted activations
                for tup in neuron.prev:
                    weighted_activ = tup[0].activation * tup[1]
                    s += weighted_activ
                s -= neuron.bias  # subtract this neuron's bias
                neuron.activation = sigmoid(s)
        # calculate the activations of all neurons in last layer
        results = []  # list of last layer activations
        for neuron in self._last_layer:
            s = 0  # initialize sum of weighted activations
            for tup in neuron.prev:
                weighted_activ = tup[0].activation * tup[1]
                s += weighted_activ
            s -= neuron.bias  # subtract this neuron's bias
            neuron.activation = sigmoid(s)
            results.append(neuron.activation)
        return results

    def _get_num_weights_and_biases(self) -> int:
        """Calculate the number of weights and biases in this network."""
        # calculate the number of weights in this network
        num_weights = len(self.first_layer) * len(self._middle_layers[0])
        for i in range(1, len(self._middle_layers)):
            curr_layer = self._middle_layers[i]
            prev_layer = self._middle_layers[i-1]
            num_weights += len(curr_layer) * len(prev_layer)
        num_weights += len(self._middle_layers[-1]) * len(self._last_layer)
        # calculate the number of biases in this network
        num_biases = 0
        for layer in self._middle_layers:
            num_biases += len(layer)
        num_biases += len(self._last_layer)
        num_weights_and_biases = num_weights + num_biases
        return num_weights_and_biases

    def _get_neuron_derivs(self, expected_results: List[float])\
        -> List[List[float]]:
        """Return a list of lists, where each list represents a layer
        of the network. Each element in a given list is the derivative
        of the network's cost function with respect to the neuron (activation)
        that element represents.

        To be used by method NeuralNetwork._get_grad *only*.
        """
        # calculate partial derivative of the cost function w.r.t. each neuron,
        # storing it in a list.
        neuron_derivatives = []
        # calculate derivatives for last layer
        curr_layer = []  # list of derivatives of cost w.r.t. neurons in the
        # last layer
        for j in range(len(self._last_layer)):
            actual = self._last_layer[j].activation
            expected = expected_results[j]
            deriv = 2 * (actual - expected)
            curr_layer.append(deriv)
        neuron_derivatives.append(curr_layer[:])
        prev_layer = curr_layer[:]
        curr_layer = []
        # calculate derivatives for middle layers
        for i in range(len(self._middle_layers), -1, -1):
            for j in range(len(self._middle_layers[i])):
                neuron = self._middle_layers[i][j]
                deriv = 0
                for k in range(len(neuron.next)):
                    tup = neuron.next[k]
                    weight = tup[1]
                    sig = sigmoid_deriv(tup[0].get_z())
                    prev_deriv = prev_layer[k]
                    deriv += weight * sig * prev_deriv
                curr_layer.append(deriv)
            neuron_derivatives.append(curr_layer[:])
            prev_layer = curr_layer[:]
            curr_layer = []
        # calculate derivatives for first layer
        for j in range(len(self.first_layer)):
            neuron = self.first_layer[j]
            deriv = 0
            for k in range(len(neuron.next)):
                tup = neuron.next[k]
                weight = tup[1]
                sig = sigmoid_deriv(tup[0].get_z())
                prev_deriv = prev_layer[k]
                deriv += weight * sig * prev_deriv
            curr_layer.append(deriv)
            neuron_derivatives.append(curr_layer[:])
            neuron_derivatives.reverse()
        return neuron_derivatives

    def _get_grad(self, expected_results: List[float]) -> List[float]:
        """Get the gradient of the current network (based on
        one training example).

        Preconditions:
            - <expected> is a valid output for this network.
        """
        # set empty list to hold derivatives
        grad = []
        nd = self._get_neuron_derivs(expected_results)
        # get derivatives of cost w.r.t. middle layer weights
        for i in range(len(self._middle_layers)):
            layer = self._middle_layers[i]
            for j in range(len(layer)):
                neuron = layer[j]
                for k in range(len(neuron.prev)):
                    # want to find dC/dw for w_jk^(i).
                    prev_neur = neuron.prev[k][0]
                    dzdw = prev_neur.activation
                    dadz = sigmoid_deriv(neuron.get_z())
                    dcda = nd[i + 1][j]
                    deriv = dzdw * dadz * dcda
                    grad.append(deriv)
        # get derivatives of cost w.r.t. last layer weights
        for j in range(len(self._last_layer)):
            neuron = self._last_layer[j]
            for k in range(len(neuron.prev)):
                prev_neur = neuron.prev[k][0]
                dzdw = prev_neur.activation
                dadz = sigmoid_deriv(neuron.get_z())
                dcda = nd[-1][j]
                deriv = dzdw * dadz * dcda
                grad.append(deriv)
        # get derivatives of cost w.r.t to middle layer biases
        for i in range(len(self._middle_layers)):
            layer = self._middle_layers[i]
            for j in range(len(layer)):
                neuron = layer[j]
                dzdw = 1  # redundant line but necessary for clarity
                dadz = sigmoid_deriv(neuron.get_z())
                dcda = nd[i + 1][j]
                deriv = dzdw * dadz * dcda
                grad.append(deriv)
        # get derivatives of cost w.r.t to last layer biases
        for j in range(len(self._last_layer)):
            neuron = self._last_layer[j]
            dzdw = 1  # redundant line but necessary for clarity
            dadz = sigmoid_deriv(neuron.get_z())
            dcda = nd[-1][j]
            deriv = dzdw * dadz * dcda
            grad.append(deriv)
        return grad

    def _get_stoch_gradient(self, data_sample: List[TrainingDataPair]) \
            -> List[float]:
        """Compute the negative stochastic gradient (i.e. average approximate
        gradient) of this network's cost using a data sample.

        Preconditions:
            - <data_sample> is a valid data sample for this neural network, see
            this library's README for more information about how training data
            is processed.
        """
        # create an placeholder list representing our gradient vector
        num_weights_and_biases = self._get_num_weights_and_biases()
        grad = [0 for _ in range(num_weights_and_biases)]
        # find the average gradient of the cost of all pairs in
        # this data sample.
        for pair in data_sample:
            self._compute_result(pair.inp)
            curr_grad = self._get_grad(pair.exp_output)
            grad = list_addition(grad, curr_grad)
        grad = list_division(grad, len(data_sample))
        grad = list_multiplication(grad, -1)
        return grad

    def _adjust_network(self, grad: List[float]) -> None:
        """Adjust this network's weights and biases based on the gradient
        vector <grad>.

        Preconditions:
            - <grad> is a valid gradient vector for this network.
             What qualifies a valid gradient vector is outlined in this
             library's README file.
        """
        # set useful variables
        len_first = len(self.first_layer)
        len_middle = len(self._middle_layers[0])  # we know middle layer must
        # contain at least one value since that is a representation invariant.
        # see class <NeuralNetwork>.
        num_middle_layers = len(self._middle_layers)
        len_last = len(self._last_layer)
        # adjust all weights
        # adjust weights between first layer and second layer
        for j in range(len_middle):
            neuron = self._middle_layers[0][j]
            for k in range(len(neuron.prev)):
                z = k + (len_first * j)
                neuron.change_connection_weight(k, grad[z])
        # adjust weights between all middle layers
        for i in range(1, num_middle_layers):
            layer = self._middle_layers[i]
            for j in range(len(layer)):
                neuron = layer[j]
                for k in range(len(neuron.prev)):
                    # # if i == 1:
                    # z = k + (len_middle * j) + (len_middle * len_first * i)
                    # # else:
                    z1 = k + (len_middle * len_first) + (len_middle * j)
                    z2 = len_middle * len_middle * (i-1)
                    z = z1 + z2
                    neuron.change_connection_weight(k, grad[z])
        # adjust weights between second last layer and last layer
        for j in range(len(self._last_layer)):
            neuron = self._last_layer[j]
            for k in range(len(neuron.prev)):
                z1 = k + (len_middle * len_first)
                z2 = ((num_middle_layers - 1) * len_middle) + (len_middle * j)
                z = z1 + z2
                neuron.change_connection_weight(k, grad[z])
        # adjust all biases
        # adjust biases of all middle layers
        b1 = (len_middle * len_first)
        b2 = ((num_middle_layers - 1) * (len_middle * len_middle))
        b3 = (len_middle * len_last)
        b = b1 + b2 + b3
        for i in range(num_middle_layers):
            layer = self._middle_layers[i]
            for j in range(len(layer)):
                neuron = layer[j]
                z = b + (len_middle * i) + j
                neuron.bias = grad[z]
        # adjust biases of last layer
        for j in range(len(self._last_layer)):
            neuron = self._last_layer[j]
            z = b + (len_middle * num_middle_layers) + j
            neuron.bias = grad[z]

    def train_network(self, training_data: List[TrainingDataPair],
                      sample_size: int = 500, rounds: int = 500) -> None:
        """Train the network using the training data in <training_data>. Each
        round of stochastic gradient calculation (called a 'sampling round'),
        a random sample of size <sample_size> is taken. There are <rounds>
        rounds.

        Preconditions:
            - <training_data> contains only training data pairs valid for this
             network.
        """
        for _ in range(rounds):  # loop through rounds
            sample = collect_sample(training_data, sample_size)
            stoch_grad = self._get_stoch_gradient(sample)
            self._adjust_network(stoch_grad)


def sigmoid(x: float) -> float:
    """Calculate the sigmoid function (also called the logistic function) of
    the parameter <x>."""
    return 1 / (1 + exp(-x))


def sigmoid_deriv(x: float) -> float:
    """Calculate the derivative of the sigmoid function as evaluated at <x>."""
    s = sigmoid(x)
    return s + (1 - s)


def list_addition(lst1: List[float], lst2: List[float]) -> List[float]:
    """Return a list whose element at an arbitrary index <i> is the sum of
    <lst1[i]> and <lst2[i]>.

    Preconditions:
        - <lst1> and <lst2> are the same size.

    >>> lst_a, lst_b = [1, 1, 2, 3, 10], [7, 9, 10, 2, 1]
    >>> list_addition(lst_a, lst_b)
    [8, 10, 12, 5, 11]
    """
    return_lst = []
    for i in range(len(lst1)):
        return_lst.append(lst1[i] + lst2[i])
    return return_lst


def list_multiplication(lst: List[float], num: float) -> List[float]:
    """Return a list whose element at an arbitrary index <i> is equal to
    <lst[i] * num>."""
    return_lst = []
    for item in lst:
        return_lst.append(item * num)
    return return_lst


def list_division(lst: List[float], num: float) -> List[float]:
    """Return a list whose element at an arbitrary index <i> is equal to
    <lst[i]> divided by <num>.
    """
    return_lst = []
    for item in lst:
        return_lst.append(item / num)
    return return_lst


if __name__ == '__main__':
    # Test to make sure Neural Network works.
    """ 
    network = NeuralNetwork(6, 2, 16, 10)
    network.init_first_layer([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    print(network.compute_result())
    """
    import doctest
    doctest.testmod()
