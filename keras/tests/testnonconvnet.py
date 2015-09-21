import unittest
import random
import numpy as np
import theano
import theano.tensor as T

from keras import models
from keras.layers import embeddings
from keras.layers import core

from nick.nonconvnet import ZeroFillDiagonals, \
        SplitOutputByFilter, \
        SlidingWindowL2Pooling

class TestNonConvNet(unittest.TestCase):
    def setUp(self):
        self.n_vocab = 100
        self.n_word_dims = 5
        self.filter_width = 4
        self.n_filters = 3
        self.max_sent_len = 9
        self.n_samples = 3

    def setSeeds(self):
        np.random.seed(1)

    def testSimpleNonConvNet(self):
        self.setSeeds()

        x = np.random.randint(self.n_vocab, size=(self.n_samples,
                self.max_sent_len))

        model = models.Sequential()

        # input: (n_samples, max_sent_len)
        # output: (n_samples, max_sent_len, n_word_dims)
        model.add(embeddings.Embedding(self.n_vocab, self.n_word_dims))
        model.compile(loss='mse', optimizer='sgd')
        expected_shape_l1 = (self.n_samples, self.max_sent_len,
                self.n_word_dims)
        output_l1 = model.predict(x)
        self.assertEqual(expected_shape_l1, output_l1.shape)

        # input: (n_samples, max_sent_len, n_word_dims)
        # output: (n_samples, max_sent_len, filter_width)
        model.add(core.TimeDistributedDense(
            self.n_word_dims, self.filter_width))
        model.compile(loss='mse', optimizer='sgd')
        expected_shape_l2 = (self.n_samples, self.max_sent_len,
                self.filter_width)
        output_l2 = model.predict(x)
        self.assertEqual(expected_shape_l2, output_l2.shape)

    def testSplitOutputByFilter(self):
        self.setSeeds()

        #n_samples = 2
        #max_sent_len = 3
        #n_filters = 4
        #filter_width = 5

        input_shape = (self.n_samples, self.max_sent_len,
                self.n_filters * self.filter_width)
        output_shape = (self.n_samples, self.n_filters,
                self.max_sent_len, self.filter_width)

        x = np.arange(np.prod(input_shape))
        x = x.reshape(input_shape).astype(np.int32)
        y = np.zeros_like(x)
        y = np.reshape(y, output_shape)

        for i in range(self.n_filters):
            s = x[:, :, i*self.filter_width:(i+1)*self.filter_width]
            y[:, i, :, :] = s

        xt = T.itensor3('xt')
        layer = SplitOutputByFilter(self.n_filters, self.filter_width)
        yt = layer._get_output(xt)

        f = theano.function(inputs=[xt], outputs=yt)

        y_theano = f(x)
        self.assertEquals(y.shape, y_theano.shape)
        self.assertTrue(np.all(y == y_theano))


    @unittest.skip("not implemented")
    def testSlidingWindowL2Pooling(self):
        self.assertTrue(
                self.max_sent_len - self.filter_width > self.n_filters)

        self.setSeeds()

        # First axis of input holds filters, second holds words,
        # and third is the span of a filter.
        input_shape = (self.n_filters, self.max_sent_len,
                self.filter_width)
        x = np.zeros(shape=input_shape)
        max_input_shape = (self.filter_width, self.filter_width)

        # For the i-th filter, make i the offset at which the maximum
        # L2 norm occurs.
        for i in np.arange(self.n_filters):
            start = i; end = i+self.filter_width
            x[i, start:end, :] = np.random.normal(size=max_input_shape)

        layer = SlidingWindowL2Pooling(self.filter_width)

        # Forward
        output = layer.forward_cpu((x,))[0]
        self.assertEquals(output.ndim, 3)
        expected_shape = (self.n_filters, self.filter_width,
                self.filter_width)
        self.assertTrue(np.allclose(output.shape, expected_shape))

        for i in np.arange(self.n_filters):
            start = i; end = i+self.filter_width
            expected = x[i, start:end, :]
            self.assertTrue(np.allclose(output[i], expected))

        # Backward -- pass the output through and expect the original input.
        grad = layer.backward_cpu((x,), (output,))[0]
        self.assertTrue(grad.ndim == x.ndim)
        self.assertTrue(np.allclose(x, grad))

    @unittest.skip("not implemented")
    def testZeroFillDiagonals(self):
        raise NotImplementedError()
