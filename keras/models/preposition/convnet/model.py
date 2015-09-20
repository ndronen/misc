import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.constraints import maxnorm
from keras.regularizers import l2
from keras.optimizers import SGD, Adam

from nick.layers import ImmutableEmbedding
from nick.difference import TemporalDifference

def build_model(args):
    print("args", vars(args))

    np.random.seed(args.seed)

    model = Sequential()

    if hasattr(args, 'weights') and args.weights is not None:
        W = np.load(args.weights)
        model.add(ImmutableEmbedding(args.n_vocab, args.n_word_dims,
            weights=[W]))
    else:
        model.add(Embedding(args.n_vocab, args.n_word_dims,
            W_constraint=maxnorm(args.embedding_max_norm)))

    print('First row of embedding weights ' + str(model.layers[0].params[0][0, :].eval()))

    if args.use_difference:
        model.add(TemporalDifference())

    model.add(Convolution1D(args.n_word_dims, args.n_filters, args.filter_width,
        W_constraint=maxnorm(args.filter_max_norm),
        border_mode='valid',
        W_regularizer=l2(args.l2_penalty)))
    model.add(Activation('relu'))

    model.add(MaxPooling1D(
        pool_length=args.input_width-args.filter_width,
        stride=None, ignore_border=True))
    model.add(Dropout(0.5))
    model.add(Flatten())

    model.add(Dense(args.n_filters, 2*args.n_filters,
        W_regularizer=l2(args.l2_penalty)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    '''
    model.add(Dense(2*args.n_filters, 2*args.n_filters))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    '''

    model.add(Dense(2*args.n_filters, args.n_filters,
        W_regularizer=l2(args.l2_penalty)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(args.n_filters, args.n_classes,
        W_regularizer=l2(args.l2_penalty)))
    model.add(Activation('softmax'))

    if args.optimizer == 'SGD':
        optimizer = SGD(lr=args.learning_rate,
            decay=args.decay, momentum=args.momentum)
    elif args.optimizer == 'Adam':
        optimizer = Adam()
    else:
        raise ValueError("don't know how to use optimizer {0}".format(args.optimizer))

    if args.loss == 'fbeta':
        loss = fbeta_builder(args.beta)
    
    model.compile(loss=args.loss, optimizer=optimizer)

    return model
