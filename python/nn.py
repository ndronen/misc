#!/usr/bin/env python

import sys
import numpy as np
import unittest

class AbstractActivation(object):
    def forward(self, x):
        """
        Computes a non-linear activation function of the pre-activation
        output of a layer of a neural network.

        Parameters
        ----------
        x : np.ndarray
            The pre-activation output of the layer.

        Returns
        ----------
        x : np.ndarray
            The post-activation output of the layer.
        """
        raise NotImplementedError()

    def gradient(self, x, cost):
        raise NotImplementedError()

class Linear(AbstractActivation):
    def forward(self, x):
        return x

    def gradient(self, x, cost):
        return x

class Sigmoid(AbstractActivation):
    def forward(self, x):
        return 2.0 / (1.0 + np.exp(-x))

    def gradient(self, x, cost):
        return self.forward(x) * (1 - self.forward(x))

class Tanh(AbstractActivation):
    def forward(self, x):
        pass

    def gradient(self, x, cost):
        pass

class Relu(AbstractActivation):
    def forward(self, x):
        return np.max(0, x)

    def gradient(self, x, cost):
        pass

class Layer(object):
    def __init__(self, W, b, activation):
        self.W = W
        self.b = b
        self.activation = activation

    def forward(self, x):
        preactivation = np.dot(self.W, x) + self.b
        self.output = self.activation.forward(preactivation)
        return self.output

    def backward(self, x, cost):
        self.grad = self.activation.gradient(x, cost)
        return self.grad

    def __str__(self):
        return ','.join([str(o) for o in [self.W, self.b, self.activation]])

class Cost(object):
    def compute(self, x, y):
        raise NotImplementedError()

class MSE(object):
    def compute(self, x, y):
        return np.mean((x - y)**2)

class TestNetwork(unittest.TestCase):
    @unittest.skip('skipping forward for now')
    def test_forward(self):
        W = np.random.normal(size=(1, 2))
        b = np.ones(shape=1)
        layer = Layer(W, b, Sigmoid())
        x = np.random.normal(size=(2, 1))
        print(x)
        print(layer.forward(x))

    #@unittest.skip('skipping backward for now')
    def test_backward(self):
        W = np.random.normal(size=(1, 3))
        b = np.ones(shape=1)
        #layer = Layer(W, b, Sigmoid())
        layer = Layer(W, b, Linear())
        
        # Create simulated regression data.
        X = np.random.normal(size=(1000,3))
        y = 1.5*X[:, 0] - 2*X[:, 1] + np.random.normal(scale=0.25, size=(1000,))

        print('W', W)
        print('b', b)
        print('y', y)

        learning_rate = 0.03

        for i in range(300):
            yhat = layer.forward(X.T)
            #print('yhat', yhat)
            cost = MSE().compute(y, yhat)
            print('cost', cost)
            #grad = layer.backward(X, cost)
            #print('backward', grad)
            update = learning_rate * np.mean(np.dot((yhat - y), X), axis=0)
            #print('update', update)
            W = W - learning_rate * np.mean(update)
            #print('W', W)
            #layer = Layer(W, b, Sigmoid())
            layer = Layer(W, b, Linear())


if __name__ == '__main__':
    unittest.main()
