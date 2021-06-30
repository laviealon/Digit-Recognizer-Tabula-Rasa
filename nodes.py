from __future__ import annotations
from typing import List, Tuple


class Neuron:
    """A node in a neural network.

    === Public Attributes ===
    activation: a node's activation.
    next: a list of tuples, where the first entry is a neuron in the next layer
        that this neuron connects to, and the second entry is the weight of
        the connect.
    prev: a list of tuples, where the first entry is a neuron in the previous
        layer that this neuron connects to before, and the second entry is
        the weight of the connect.

    === Representation Invariants ===
    - a neuron's activation must be between 0 and 1.
    - all weights must be between 0 and 1.
    """
    activation: float
    next: List[Tuple[Neuron, float]]
    prev: List[Tuple[Neuron, float]]

    def __init__(self, activation: float) -> None:
        """Insert docstring."""
        self.activation = activation
        self.next = []
        self.prev = []

    def add_nextlayer_connection(self, neuron: Neuron, weight: float) -> None:
        """Insert docstring."""
        tup = neuron, weight
        self.next.append(tup)
        neuron.prev.append((self, weight))


class NeuralNetwork:
    """A Neural Network.

    === Public Attributes ===
    first_layer: a list of the neurons in this network's first layer.
    """
    first_layer: List[Neuron]

    def __init__(self, *args: Neuron) -> None:
        """Insert docstring."""
        neurons = []
        for arg in args:
            neurons.append(arg)
        self.first_layer = neurons


if __name__ == '__main__':
    pass
