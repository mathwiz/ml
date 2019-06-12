import unittest
import numpy as np
from neuron import *


def even(x):
    return x % 2 == 0


def expt_inv(b, p, a):
    if p == 0:
        return a
    elif even(p):
        return expt_inv(b * b, p / 2, a)
    else:
        return expt_inv(b, p - 1, a * b)


def expt(b, p):
    return expt_inv(b, p, 1)


class MyTestCase(unittest.TestCase):
    def test1(self):
        self.assertEqual(ReLU(1), 1)
        self.assertEqual(ReLU(0), 0)
        self.assertEqual(ReLU(-1), 0)

    def test2(self):
        n = Neuron(ReLU)
        self.assertEqual(n.compute(np.array([.1, .1, 0])), 0.2)
        self.assertEqual(n.compute(np.array([.1, .1, -.3])), 0)



if __name__ == '__main__':
    unittest.main()


