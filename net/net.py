import numpy as np
import six

def sigmoid(x):
    """
    >>> import matplotlib as plt
    >>> import numpy as np
    >>> values = np.linspace(-10, 10, 99)
    >>> plt.plot(values, sigmoid(values))
    """
    return 1. / (1. + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def identity(x):
    return x

def softmax(z):
    return np.exp(z)/np.sum(np.exp(z))

def normalized(z):
    """
    To get a sense of how the softmax function differs from simple
    normalization, run
    >> softmax([0, -1])
    >> normalized([0, -1])
    """
    return z/np.sum(z)

class Layer(object):
    def __init__(self, activation, nvis, nhid, learning_rate):
        self.activation = activation
        self.W = np.random.uniform(0, 0.1, size=(nvis, nhid))
        self.lr = learning_rate
        self.output = None

    def forward(self, x):
        #self.output = self.activation(np.dot(x, self.W) + self.b)
        self.output = self.activation(np.dot(x, self.W))
        return self.output

    '''
    def backward(self, y, cost):
        error = cost(y, self.output)
        self.W = self.W - self.lr * self.gradient(cost)
    '''
