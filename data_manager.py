"""This file is responsible for the management of transformed image data
so that it can be loaded into a neural network created by <network.py>."""


from typing import List


class TrainingDataPair:
    inp: List[float]
    exp_output: List[float]

    def __init__(self, inp: List[float], outp: List[float]) -> None:
        self.inp = inp
        self.exp_output = outp


if __name__ == '__main__':
    pass