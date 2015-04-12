#!/usr/bin/env python

import sys
import numpy as np
import unittest

class AbstractActivation(object):
    def forward(self, x):
        raise NotImplementedError()

    def gradient(self, cost):
        raise NotImplementedError()

class Linear(AbstractActivation):
    def forward(self, x):
        return np.sum(x)

    def gradient(self, cost):
        pass

class Sigmoid(AbstractActivation):
    def forward(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def gradient(self, cost):
        pass

class Tanh(AbstractActivation):
    def forward(self, x):
        pass

    def gradient(self, cost):
        pass

class Relu(AbstractActivation):
    def forward(self, x):
        return np.max(0, x)

    def gradient(self, cost):
        pass

class Layer(object):
    def __init__(self, W, b, activation):
        self.W = W
        self.b = b
        self.activation = activation

    def forward(self, x):
        self.forward = self.activation.forward(np.dot(self.W, x) + self.b)
        return self.forward

    def gradient(self, cost):
        # Here use self.forward and cost to update the weights.
        # self.activation.gradient(cost)
        pass

    def __str__(self):
        return ','.join([str(o) for o in [self.W, self.b, self.activation]])

class TestNetwork(unittest.TestCase):
    def test_create_network(self):
        W = np.random.normal(size=(1, 2))
        b = np.ones(shape=1)
        layer = Layer(W, b, Sigmoid())
        x = np.random.normal(size=(2, 1))
        layer.forward(x)

if __name__ == '__main__':
    unittest.main()
