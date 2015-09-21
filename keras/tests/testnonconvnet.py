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
        SlidingWindowL2MaxPooling

class TestNonConvNet(unittest.TestCase):
    def setUp(self):
        self.n_vocab = 100
        self.n_word_dims = 5
        self.filter_width = 4
        self.n_filters = 3
        self.max_seq_len = 9
        self.n_samples = 3

    def setSeeds(self):
        np.random.seed(1)

    @unittest.skip('disabled temporarily')
    def testSimpleNonConvNet(self):
        self.setSeeds()

        x = np.random.randint(self.n_vocab, size=(self.n_samples,
                self.max_seq_len))

        model = models.Sequential()

        # input: (n_samples, max_seq_len)
        # output: (n_samples, max_seq_len, n_word_dims)
        model.add(embeddings.Embedding(self.n_vocab, self.n_word_dims))
        model.compile(loss='mse', optimizer='sgd')
        expected_shape_l1 = (self.n_samples, self.max_seq_len,
                self.n_word_dims)
        output_l1 = model.predict(x)
        self.assertEqual(expected_shape_l1, output_l1.shape)

        # input: (n_samples, max_seq_len, n_word_dims)
        # output: (n_samples, max_seq_len, filter_width)
        model.add(core.TimeDistributedDense(
            self.n_word_dims, self.filter_width))
        model.compile(loss='mse', optimizer='sgd')
        expected_shape_l2 = (self.n_samples, self.max_seq_len,
                self.filter_width)
        output_l2 = model.predict(x)
        self.assertEqual(expected_shape_l2, output_l2.shape)

    #@unittest.skip('disabled temporarily')
    def testSplitOutputByFilter(self):
        self.setSeeds()

        input_shape = (self.n_samples, self.max_seq_len,
                self.n_filters * self.filter_width)
        output_shape = (self.n_samples, self.n_filters,
                self.max_seq_len, self.filter_width)

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

    def testSlidingWindowL2MaxPooling(self):
        self.assertTrue(
                self.max_seq_len - self.filter_width > self.n_filters)

        self.setSeeds()

        input_shape = (self.n_samples, self.n_filters,
                self.max_seq_len, self.filter_width)
        output_shape = (self.n_samples, self.n_filters,
                self.filter_width, self.filter_width)

        x = np.zeros(shape=input_shape)

        max_input_shape = (self.n_samples, self.filter_width, self.filter_width)

        # For the i-th filter, make i the offset at which the maximum
        # L2 norm occurs.
        for i in np.arange(self.n_filters):
            start = i
            end = i+self.filter_width
            values = i + np.arange(np.prod(max_input_shape))
            x[:, i, start:end, :] = values.reshape(max_input_shape)

        it = T.iscalar()
        x3d = T.dtensor3('x3d')
        x4d = T.dtensor4('x4d')

        layer = SlidingWindowL2MaxPooling(
                self.n_samples, self.n_filters, self.filter_width,
                self.max_seq_len)

        '''
        Use the first sample and first filter to test `filter_dimension`.
        '''
        yt_filter_dim = layer.filter_dimension(it, x3d)
        f_filter_dim = theano.function(inputs=[it, x3d], outputs=yt_filter_dim)
        y_filter_dim_out = f_filter_dim(0, x[0])
        expected = x[0, 0, 0:4, :]
        self.assertEquals((self.filter_width, self.filter_width),
                y_filter_dim_out.shape)
        self.assertTrue(np.all(expected == y_filter_dim_out))

        '''
        Use the first sample to test `filter_dimension`.
        '''
        yt_sample_dim = layer.sample_dimension(it, x4d)
        f_sample_dim = theano.function(inputs=[it, x4d], outputs=yt_sample_dim)
        y_sample_dim_out = f_sample_dim(0, x)
        expected = x[0:1, :, 0:4, :] 
        expected[0, 0, :, :] = x[0, 0, 0:4, :]
        expected[0, 1, :, :] = x[0, 1, 1:5, :]
        expected[0, 2, :, :] = x[0, 2, 2:6, :]
        self.assertEquals((self.n_filters, self.filter_width, self.filter_width),
                y_sample_dim_out.shape)
        self.assertTrue(np.all(expected == y_sample_dim_out))

    @unittest.skip("not implemented")
    def testZeroFillDiagonals(self):
        raise NotImplementedError()
