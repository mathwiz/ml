import numpy as np

def ReLU(x):
    return max(0, x)


class Neuron:
    def __init__(self, activation=ReLU):
        self.activation = activation


    def compute(self, inputs):
        return self.activation(sum(inputs))


