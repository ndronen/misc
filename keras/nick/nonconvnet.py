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
    input: (n_samples, max_sent_len, n_filters * filter_width)
    output: (n_samples, n_filters, max_sent_len, filter_width)
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
        outputs, updates = theano.scan(fn=self.slice,
            outputs_info=None,
            sequences=[T.arange(self.n_filters)],
            non_sequences=X)
        return outputs.dimshuffle(1, 0, 2, 3)

    def get_output(self, train):
        return self._get_output(self.get_input(train))

    def get_config(self):
        return {"name": self.__class__.__name__}

class SlidingWindowL2Pooling(Layer):
    def __init__(self, filter_width):
        self.filter_width = filter_width

    def norm_in_window(self, x, i):
        return x[i:i+self.filter_width, :].sum()

    def forward_cpu(self, x):
        # The output of this layer is a 3-tensor with filters on the first
        # dimension and the second and third each being filter width.
        output_shape=(x[0].shape[0], self.filter_width, self.filter_width)
        output = np.zeros(shape=output_shape, dtype=np.float32)

        self.indices = np.zeros(x[0].shape[0])
        start_indices = np.arange(x[0][0, :, :].shape[0] - self.filter_width)

        for i in np.arange(output_shape[0]):
            x_i = x[0][i, :, :]
            x2 = x_i**2
            norms = [self.norm_in_window(x2, j) for j in start_indices]
            max_norm_index = np.argmax(norms)
            self.indices[i] = max_norm_index
            output[i] = x_i[max_norm_index:max_norm_index+self.filter_width, :]

        logger.debug([str(val) for val in [x[0].shape, output.shape]])
        return output,

class ZeroFillDiagonals(Layer):
    def forward_cpu(self, x):
        for i in np.arange(x[0].shape[0]):
            np.fill_diagonal(x[0][i, :, :], 0)
        logger.debug(x[0].shape)
        return x
