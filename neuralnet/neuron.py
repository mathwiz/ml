import numpy as np


def ReLU(x):
    return max(0, x)


class Neuron:
    def __init__(self, activation=ReLU):
        self.activation = activation
        self.inputs = []

    def compute(self, inputs):
        return self.activation(sum(inputs))

    def add(self, neuron):
        self.inputs.append(neuron)
        return self

    def output(self):
        pass

