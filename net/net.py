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

nvis = 2
nout = 1

# Simple linear regression.
model = Layer(activation=identity, nvis=nvis, nhid=nout, learning_rate=0.1)

np.random.seed(17)
n_examples = 1000
x = np.random.normal(size=(n_examples,nvis))
y = (x[:, 0] - 2*x[:, 1]).reshape((n_examples,nout))
y += np.random.uniform(-0.05, 0.05, size=y.shape)

for epoch in six.moves.range(1000):
    y_hat = model.forward(x)

    assert y.shape == y_hat.shape

    # Hard-coded cost function.
    mse = np.sum((y - y_hat)**2)/len(x)

    # Hard-coded gradient computation.
    grad = np.dot(x.T, y_hat - y)

    # Updating parameters outside of the layer.
    model.W = model.W - model.lr * grad/float(len(x))

print(np.concatenate([y, y_hat], axis=1))
