#!/usr/bin/env python

import model

class MyClassifier(model.Classifier):
    def init_params(self):
        self.params = chainer.FunctionSet(
            l1=F.Linear(self.n_inputs, 2*self.n_units),
            l2=F.Linear(2*self.n_units, 2*self.n_units),
            l3=F.Linear(2*self.n_units, self.n_classes))

    def forward(self, data, train=True):
        x = chainer.Variable(data)
        h1 = F.dropout(F.relu(self.params.l1(x)), train=train)
        h2 = F.dropout(F.relu(self.params.l2(h1)), train=train)
        return self.params.l3(h2)

import argparse
import copy
import cPickle
import marshal

import numpy as np
import pandas as pd
import six

import chainer
from chainer import computational_graph as c
from chainer import cuda
import chainer.functions as F
from chainer import optimizers

parser = argparse.ArgumentParser(description='Train a model')
parser.add_argument('--train-size', type=int,
        help='Train only using this many examples from the training set.')
parser.add_argument('--model-path', type=str,
        help='Path to which to save best model')
parser.add_argument('--gpu', '-g', default=-1, type=int,
        help='GPU ID (negative value indicates CPU)')
parser.add_argument('--batch-size', default=128, type=int, dest="batch_size",
        help='The minibatch size')
parser.add_argument('--n-units', default=200, type=int, 
        help='The (effective) number of units in a hidden layer')
parser.add_argument('--learning-rate', default=0.1, type=float,
        help='The learning rate for (minibatch) SGD')
parser.add_argument('--learning-rate-decay', type=float, default=0.025,
        help='Learning rate decay; learning rate is reduced by this fraction every epoch')
parser.add_argument('--learning-rate-min', type=float, default=0.0001,
        help='When using learning rate decay, the minimum learning rate')
parser.add_argument('--momentum', default=0., type=float,
        help='The momentum for (minibatch) SGD')
parser.add_argument('--weight-decay', default=0.0, type=float,
        help='The weight decay value.  Disabled by default (0.0)')
parser.add_argument('--patience', type=int, default=10,
        help='Number of epochs to wait for improved performance before stopping')
parser.add_argument('--n-epochs', type=int, default=None,
        help='The maximum number of epochs to run')
parser.add_argument('--optimizer', type=str, default='SGD',
        choices=['SGD', 'AdaDelta', 'AdaGrad', 'Adam', 'RMSprop'],
        help='The maximum number of epochs to run')

args = parser.parse_args()

if args.gpu >= 0:
    cuda.check_cuda_available()
xp = cuda.cupy if args.gpu >= 0 else np

raise NotImplementedError('implement data loading')

'''
train_vectors, train_scores, \
    validation_vectors, validation_scores, \
    test_vectors, test_scores = data.load_aspire_data(args.prompt,
            analyzer=args.analyzer, ngram_range=args.ngram_range,
            max_features=args.max_features)
'''

if args.train_size < train_vectors.shape[0]:
    # Take the first `train_size` rows.
    train_vectors = train_vectors[0:args.train_size, :]
    train_scores = train_scores[0:args.train_size]

max_features = min(args.max_features, train_vectors.shape[1])

print('args', args)
print('train_vectors', train_vectors.shape)
print('validation_vectors', validation_vectors.shape)
print('test_vectors', test_vectors.shape)
print('n_features', max_features)

N = len(train_scores)
N_validation = len(validation_scores)
N_test = len(test_scores)

model = Model(args)

if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()

x_train = train_vectors
y_train = train_scores
x_validation = validation_vectors
y_validation = validation_scores
x_test = test_vectors
y_test = test_scores

best_accuracy = 0.
best_epoch = 0

def keep_training(epoch, best_epoch):
    if args.n_epochs is not None and epoch > args.n_epochs:
            return False
    if epoch > 1 and epoch - best_epoch > args.patience:
        return False
    return True

epoch = 1

while True:
    if not keep_training(epoch, best_epoch):
        break

    # training
    if args.fixed_batch_order:
        perm = np.arange(N)
    else:
        perm = np.random.permutation(N)

    sum_accuracy = 0
    sum_loss = 0
    for j, i in enumerate(six.moves.range(0, N, args.batch_size)):
        x_batch = xp.asarray(x_train[perm[i:i + args.batch_size]])
        y_batch = xp.asarray(y_train[perm[i:i + args.batch_size]])
        pred, loss, acc = model.fit(x_batch, y_batch)
        sum_loss += float(loss.data) * len(y_batch)
        sum_accuracy += float(acc.data) * len(y_batch)
    print('train epoch={}, mean loss={}, accuracy={}'.format(
        epoch, sum_loss / N, sum_accuracy / N))

    '''
    if args.learning_rate_decay > 0.:
        lr = args.learning_rate
        lr = lr * (1 - args.learning_rate_decay)
        if lr < args.learning_rate_min:
            lr = args.learning_rate_min
        optimizer = optimizers.MomentumSGD(
                lr=lr, momentum=args.momentum)
        optimizer.setup(model)
    '''

    # validation set evaluation
    sum_accuracy = 0
    sum_loss = 0
    for i in six.moves.range(0, N_validation, args.batch_size):
        x_batch = xp.asarray(x_validation[i:i + args.batch_size])
        y_batch = xp.asarray(y_validation[i:i + args.batch_size])
        pred, loss, acc = model.predict(x_batch, target=y_batch)
        sum_loss += float(loss.data) * len(y_batch)
        sum_accuracy += float(acc.data) * len(y_batch)

    validation_accuracy = sum_accuracy / N_validation
    validation_loss = sum_loss / N_validation

    if validation_accuracy > best_accuracy:
        best_accuracy = validation_accuracy
        best_epoch = epoch
        if args.model_path is not None:
            if args.gpu >= 0:
                model.to_cpu()
            store = {
                    'args': args,
                    'model': model,
                    'optimizer': optimizer
                }
            cPickle.dump(store, open(args.model_path + '.store', 'w'))
            if args.gpu >= 0:
                model.to_gpu()
            marshal.dump(forward.func_code, open(args.model_path + '.code', 'w'))

    print('validation epoch={}, mean loss={}, accuracy={} best=[accuracy={} epoch={}]'.format(
        epoch, validation_loss, validation_accuracy, 
        best_accuracy,
        best_epoch))

    # Only run the test set if our validation set performance in this
    # epoch is the best so far.
    if args.run_test and best_epoch == epoch:
        # test set evaluation
        sum_accuracy = 0
        sum_loss = 0
        df_test = pd.DataFrame({ 'score': test_scores })
        df_test['pred'] = np.nan

        for i in six.moves.range(0, N_test, args.batch_size):
            x_batch = xp.asarray(x_test[i:i + args.batch_size])
            y_batch = xp.asarray(y_test[i:i + args.batch_size])

            y_hat, loss, acc = forward(x_batch, y_batch, train=False)
    
            df_test.ix[i:i+args.batch_size-1, 'pred'] = np.argmax(y_hat.data, axis=1)
            sum_loss += float(loss.data) * len(y_batch)
            sum_accuracy += float(acc.data) * len(y_batch)

        df_test.to_csv(args.save_test, index=False)

        print('test  mean loss={}, accuracy={}'.format(
            sum_loss / N_test, sum_accuracy / N_test))

    epoch += 1
