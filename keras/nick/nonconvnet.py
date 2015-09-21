import math
import numpy as np
import theano
import theano.tensor as T
import unittest
import logging

logger = logging.getLogger()

from keras.layers.core import Layer
from keras.utils.theano_utils import sharedX

class SplitOutputByFilter(Layer):
    """
    input: (n_samples, max_seq_len, n_filters * filter_width)
    output: (n_samples, n_filters, max_seq_len, filter_width)
    """
    def __init__(self, n_filters, filter_width):
        super(SplitOutputByFilter, self).__init__()
        self.n_filters = n_filters
        self.filter_width = filter_width
        self.input = T.tensor3()

    def slice(self, i, X):
        start = i * self.filter_width
        end = (i+1) * self.filter_width
        return X[:, :, start:end]

    def _get_output(self, X):
        outputs, updates = theano.scan(
                fn=self.slice,
                outputs_info=None,
                sequences=[T.arange(self.n_filters)],
                non_sequences=X)
        return outputs.dimshuffle(1, 0, 2, 3)

    def get_output(self, train):
        return self._get_output(self.get_input(train))

    def get_config(self):
        return {"name": self.__class__.__name__}

class SlidingWindowL2MaxPooling(Layer):
    '''
    input: (n_samples, n_filters, max_seq_len, filter_width)
    output: (n_samples, n_filters, filter_width, filter_width)
    '''
    def __init__(self, batch_size, n_filters, filter_width, max_seq_len):
        self.batch_size = batch_size
        self.n_filters = n_filters
        self.filter_width = filter_width
        self.max_seq_len = max_seq_len

    def get_output(self, train):
        return self._get_output(self.get_input(train))

    def _get_output(self, X):
        outputs, updates = theano.scan(
                fn=self.sample_dimension,
                sequences=[T.arange(self.batch_size)],
                non_sequences=X)
        return outputs

    def sample_dimension(self, i, X):
        '''
        Takes a 4-tensor of shape `(n_samples, n_filters, max_seq_len,
        filter_width)` and an index into its first dimension.  Returns the
        `(n_samples, n_filters, filter_width, filter_width)` subtensor
        with the greatest L2 norm along the third dimension.

        Parameters
        ----------
        X : a 4-tensor
            An `(n_samples, n_filters, max_seq_len, filter_width)` tensor.
        i : int
            An index into the first dimension of `X`.

        Returns
        ----------
        A 3-tensor of shape `(n_filters, filter_width, filter_width)`
        consisting of the subtensor of `X` with the greatest L2 norm along
        `X`'s third dimension (where `max_seq_len` lies).
        '''
        outputs, updates = theano.scan(
                fn=self.filter_dimension,
                sequences=[T.arange(self.n_filters)],
                non_sequences=X[i, :, :, :])

        return outputs

    def filter_dimension(self, i, X):
        '''
        Takes a 3-tensor of shape `(n_filters, max_seq_len, filter_width)`
        and an index into its first dimension.  Returns the
        `(filter_width, filter_width)` subtensor of `X` with the greatest
        L2 norm along the second dimension.

        Parameters
        ----------
        X : a 3-tensor
            An `(n_samples, n_filters, max_seq_len, filter_width)` tensor.
        i : int
            An index into the first dimension of `X`.

        Returns
        ----------
        A 2-tensor of shape `(filter_width, filter_width)` consisting
        of the subtensor of the i-th element along the first dimension
        of `X` with the greatest L2 norm along `X`'s second dimension
        (where `max_seq_len` lies).
        '''
        norms, updates = theano.scan(
                fn=self.norm,
                sequences=[T.arange(self.max_seq_len)],
                non_sequences=X[i, :, :])
        start_window = T.argmax(norms)
        end_window = start_window + self.filter_width
        return X[i, start_window:end_window, :]

    def norm(self, i, X):
        return (X[i:i+self.filter_width, :] ** 2).sum()

class ZeroFillDiagonals(Layer):
    def forward_cpu(self, x):
        for i in np.arange(x[0].shape[0]):
            np.fill_diagonal(x[0][i, :, :], 0)
        logger.debug(x[0].shape)
        return x
