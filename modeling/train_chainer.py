#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import print_function

import sys
import six
import argparse
import copy
import cPickle
import marshal

import numpy as np
import pandas as pd

import chainer
from chainer import cuda
from modeling.chainer_model import Classifier
from modeling.utils import load_model_data, ModelConfig
import modeling.parser

def main(args):
    if args.gpu >= 0:
        cuda.check_cuda_available()
    xp = cuda.cupy if args.gpu >= 0 else np

    model_id = build_model_id(args)
    model_path = build_model_path(args, model_id)
    setup_model_dir(args, model_path)
    sys.stdout, sys.stderr = setup_logging(args)

    x_train, y_train = load_model_data(args.train_file,
            args.data_name, args.target_name,
            n=args.n_train)
    x_validation, y_validation = load_model_data(
            args.validation_file,
            args.data_name, args.target_name,
            n=args.n_validation)

    rng = np.random.RandomState(args.seed)

    N = len(x_train)
    N_validation = len(x_validation)

    sys.path.append(args.model_dir)
    from model import Model
    model_cfg = ModelConfig(**json_cfg)
    model = build_model(model_cfg)
    setattr(model, 'stop_training', False)
    model = Model(args)
    
    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        model.to_gpu()
    
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
    
if __name__ == '__main__':
    parser = modeling.parser.build_chainer()
    sys.exit(main(parser.parse_args()))
