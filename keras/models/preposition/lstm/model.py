import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.recurrent import LSTM, GRU
from keras.layers.embeddings import Embedding
from keras.constraints import maxnorm
from keras.optimizers import SGD, Adam, RMSprop, Adadelta, Adagrad

from nick.layers import ImmutableEmbedding

def build_model(args):
    print("args", vars(args))

    model = Sequential()

    if hasattr(args, 'weights') and args.weights is not None:
        W = np.load(args.weights)
        model.add(ImmutableEmbedding(args.n_vocab, args.n_word_dims,
            mask_zero=args.mask_zero,
            weights=[W]))
    else:
        model.add(Embedding(args.n_vocab, args.n_word_dims,
            mask_zero=args.mask_zero,
            W_constraint=maxnorm(args.embedding_max_norm)))

    model.add(LSTM(args.n_word_dims, args.n_units,
        truncate_gradient=args.truncate_gradient,
        return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(args.n_units, args.n_units,
        truncate_gradient=args.truncate_gradient,
        return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(args.n_units, args.n_units,
        truncate_gradient=args.truncate_gradient,
        return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(args.n_units, args.n_units,
        truncate_gradient=args.truncate_gradient,
        return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(args.n_units, args.n_classes))
    model.add(Activation('softmax'))

    if args.optimizer == 'SGD':
        optimizer = SGD(lr=args.learning_rate,
            decay=args.decay, momentum=args.momentum,
            clipnorm=args.clipnorm)
    elif args.optimizer == 'Adam':
        optimizer = Adam(clipnorm=args.clipnorm)
    elif args.optimizer == 'RMSprop':
        optimizer = RMSprop(clipnorm=args.clipnorm)
    elif args.optimizer == 'Adadelta':
        optimizer = Adadelta(clipnorm=args.clipnorm)
    elif args.optimizer == 'Adagrad':
        optimizer = Adagrad(clipnorm=args.clipnorm)
    else:
        raise ValueError("don't know how to use optimizer {0}".format(args.optimizer))

    setattr(model, 'stop_training', False)

    model.compile(loss=args.loss, optimizer=optimizer)

    return model
