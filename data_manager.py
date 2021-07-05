"""This file is responsible for the management of transformed image data
so that it can be loaded into a neural network created by <network.py>."""

from random import choices
from typing import List


class TrainingDataPair:
    inp: List[float]
    exp_output: List[float]

    def __init__(self, inp: List[float], outp: List[float]) -> None:
        self.inp = inp
        self.exp_output = outp


def collect_sample(pop: List[TrainingDataPair], sample_size: int) \
                    -> List[TrainingDataPair]:
    """Collect a random sample of size <sample_size> from <pop>."""
    return choices(pop, k=sample_size)


if __name__ == '__main__':
    pass
