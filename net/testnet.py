import numpy as np
import six
import net
import unittest

class TestNet(unittest.TestCase):
    def test_linear_regression(self):
        nvis = 2
        nout = 1

        # Simple linear regression.
        model = net.Layer(activation=net.identity,
                nvis=nvis, nhid=nout, learning_rate=0.1)

        np.random.seed(17)
        n_examples = 1000
        x = np.random.normal(size=(n_examples,nvis))
        y = (x[:, 0] - 2*x[:, 1]).reshape((n_examples,nout))

        for epoch in six.moves.range(1000):
            y_hat = model.forward(x)

            assert y.shape == y_hat.shape

            # Hard-coded cost function.
            mse = np.sum((y - y_hat)**2)/len(x)

            # Hard-coded gradient computation.
            grad = np.dot(x.T, y_hat - y)

            # Updating parameters outside of the layer.
            model.W = model.W - model.lr * grad/float(len(x))

        self.assertTrue(np.allclose(y, y_hat))

if __name__ == '__main__':
    unittest.main()
