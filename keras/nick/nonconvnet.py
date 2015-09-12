import math
import numpy as np
import theano.tensor as T
import unittest
import logging

logger = logging.getLogger()

from keras.layers.core import Layer

class SplitOutputByFilter(Layer):
    """
    input: (n_samples, max_sent_len, n_filters * filter_width)
    output: (n_samples, max_sent_len, n_filters, filter_width)
    """
    def __init__(self, n_filters):
        super(SplitOutputByFilter, self).__init__()
        self.n_filters = n_filters
        self.input = T.tensor3()

    def _get_output(self, X):
        # x = vector()
        # splits = lvector()
        # You have to declare, in advance, the number of splits.
        #points = T.repeat(X.shape)
        splits = T.split(X, splits, n_splits=self.n_filters, axis=2)

        # f = function([x, splits], [ra, rb, rc])
        # a, b, c = f([0,1,2,3,4,5], [3, 2, 1])
        # a == [0,1,2]
        # b == [3, 4]
        # c == [5]

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
