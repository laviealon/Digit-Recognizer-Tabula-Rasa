"""This file is responsible for the management of transformed image data
so that it can be loaded into a neural network created by <network.py>."""
import numpy as np
from random import choices
from typing import List


class DataPair:
    inp: np.array
    exp_output: np.array

    def __init__(self, inp: np.array, outp: np.array) -> None:
        self.inp = inp
        self.exp_output = outp


def collect_sample(pop: List[DataPair], sample_size: int) \
                    -> List[DataPair]:
    """Collect a random sample of size <sample_size> from <pop>."""
    return choices(pop, k=sample_size)


if __name__ == '__main__':
    pass
