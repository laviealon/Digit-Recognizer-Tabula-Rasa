from __future__ import annotations
from typing import List, Tuple, Optional
from training_data_pair import TrainingDataPair
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

    def _init_first_layer(self, activations: List[float]) -> None:
        """Insert docstring.

        Preconditions:
            - len(activations) is equal to len(self.first_layer).
            - all floats in <activations> are between 0 and 1.
        """
        for i in range(len(self.first_layer)):
            activation = activations[i]
            neuron = self.first_layer[i]
            neuron.activation = activation

    def _compute_result(self) -> List[float]:
        """ Compute the output of this neural network whose first
        layer has already been initialized.

        Preconditions:
            - init_first_layer has already been called on this network.
        """
        for layer in self._middle_layers:
            for neuron in layer:
                s = 0  # initialize sum of weighted activations
                for tup in neuron.prev:
                    weighted_activ = tup[0].activation * tup[1]
                    s += weighted_activ
                s -= neuron.bias  # subtract this neuron's bias
                neuron.activation = sigmoid(s)
        results = []
        for neuron in self._last_layer:
            s = 0  # initialize sum of weighted activations
            for tup in neuron.prev:
                weighted_activ = tup[0].activation * tup[1]
                s += weighted_activ
            s -= neuron.bias  # subtract this neuron's bias
            neuron.activation = sigmoid(s)
            results.append(neuron.activation)
        return results

    def _costof_one_pair(self, data_pair: TrainingDataPair) -> float:
        """Compute the cost of this network's performance of
        this data pair.

        Preconditions:
            - data_pair is a valid data pair for this neural network.
        """
        pass




def sigmoid(x: float) -> float:
    return 1 / (1 + exp(-x))


if __name__ == '__main__':
    # Test to make sure Neural Network works.
    """ 
    network = NeuralNetwork(6, 2, 16, 10)
    network.init_first_layer([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    print(network.compute_result())
    """
    pass
