#!/usr/bin/env python

import os, sys, shutil
import argparse
import logging
import json
import uuid
import json
import itertools 

import numpy as np
import theano
import h5py
import six
from sklearn.metrics import classification_report, fbeta_score


from keras.utils import np_utils
from keras.optimizers import SGD
import keras.callbacks
from keras.callbacks import ModelCheckpoint, EarlyStopping
import keras.models

sys.path.append('.')

from nick.callbacks import ClassificationReport, OptimizerMonitor
from nick.utils import (count_parameters, callable_print,
        ModelConfig, LoggerWriter)

def load_data(path, data_name, target_name):
    hdf5 = h5py.File(path)
    datasets = [hdf5[d].value.astype(np.int32) for d in data_name]
    for i,d in enumerate(datasets):
        if d.ndim == 1:
            datasets[i] = d.reshape((d.shape[0], 1))
    print([d.shape for d in datasets])
    data = np.concatenate(datasets, axis=1)
    target = hdf5[target_name].value.astype(np.int32)
    return data, target

"""
Train a model on sentences.
"""

def get_parser():
    parser = argparse.ArgumentParser(
            description='Train a convnet on sentences.')
    parser.add_argument('model_dir', metavar="MODEL_DIR", type=str,
            help='The base directory of this model.  Must contain a model.py (model code) and a model.json (hyperparameters).  Model configuration and weights are saved to model_dir/UUID.')
    parser.add_argument('train_file', metavar='TRAIN_FILE', type=str,
            help='HDF5 file of training examples.')
    parser.add_argument('validation_file', metavar='VALIDATION_FILE', type=str,
            help='HDF5 file of validation examples.')
    parser.add_argument('data_name', nargs='+', type=str,
            help='Name(s) of the data variable(s) in input HDF5 file.')

    parser.add_argument('--target-name', default='target_code', type=str,
            help='Name of the target variable in input HDF5 file.')
    parser.add_argument('--extra-train-file', type=str, nargs='+',
            help='path to one or more extra train files, useful for when training set is too big to fit into memory.')
    parser.add_argument('--model-dest', type=str,
            help='Directory to which to copy model.py and model.json.  This overrides copying to model_dir/UUID.')
    parser.add_argument('--target-data', type=str,
            help='Pickled dictionary of target data from sklearn.preprocessing.LabelEncoder.  The dictionary must contain a key `TARGET_NAME` that maps either to a list of target names or a dictionary mapping target names to their class weights (useful for imbalanced data sets')
    parser.add_argument('--description', type=str,
            help='Short description of this model (data, hyperparameters, etc.)')
    parser.add_argument('--seed', default=1, type=int,
            help='The seed for the random number generator')
    parser.add_argument('--shuffle', default=False, action='store_true',
            help='Shuffle the data in each minibatch')
    parser.add_argument('--n-epochs', type=int, default=100,
            help='The maximum number of epochs to train')
    parser.add_argument('--n-train', default=np.inf, type=int,
            help='The number of training examples to use')
    parser.add_argument('--n-validation', default=np.inf, type=int,
            help='The number of validation examples to use')
    parser.add_argument('--n-vocab', default=-1, type=int,
            help='The number of words in the training vocabulary')
    parser.add_argument('--n-classes', default=-1, type=int,
            help='The number of classes in TARGET_NAME')
    parser.add_argument('--log', action='store_true',
            help='Whether to send console output to log file')
    parser.add_argument('--no-save', action='store_true',
            help='Disable saving/copying of model.py and model.json to a unique directory for reproducibility')
    parser.add_argument('--classification-report', action='store_true',
            help='Include an sklearn classification report on the validation set at end of each epoch')
    parser.add_argument('--error-classes-only', action='store_true',
            help='Only report on error classes in classification report')

    return parser.parse_args()

def main(args):
    if args.model_dest:
        model_id = args.model_dest
    else:
        model_id = uuid.uuid1().hex

    model_path = args.model_dir + '/' + model_id + '/'

    if not args.no_save:
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        print("model path is " + model_path)

    if args.log and not args.no_save:
        logging.basicConfig(filename=model_path + 'model.log',
                format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                datefmt='%m-%d %H:%M',
                level=logging.DEBUG)
        sys.stdout = LoggerWriter(logging.info)
        sys.stderr = LoggerWriter(logging.warning)
    else:
        logging.basicConfig(level=logging.DEBUG)

    x_train, y_train = load_data(args.train_file,
            args.data_name, args.target_name)
    x_validation, y_validation = load_data(
            args.validation_file,
            args.data_name, args.target_name)

    if len(y_train) > args.n_train:
        logging.info("Reducing training set size from " +
                str(len(y_train)) + " to " + str(args.n_train))
        y_train = y_train[0:args.n_train]
        x_train = x_train[0:args.n_train, :]

    if len(y_validation) > args.n_validation:
        logging.info("Reducing validation set size from " +
                str(len(y_validation)) + " to " + str(args.n_validation))
        y_validation = y_validation[0:args.n_validation]
        x_validation = x_validation[0:args.n_validation, :]
    
    rng = np.random.RandomState(args.seed)

    if args.target_data:
        target_names_dict = json.load(open(args.target_data))

        try:
            target_data = target_names_dict[args.target_name]
        except KeyError:
            raise ValueError("Invalid key " + args.target_name +
                    " for dictionary in " + args.target_data)

        if isinstance(target_data, dict):
            try:
                target_names = target_data['names']
                class_weight = target_data['weights']
            except KeyError, e:
                raise ValueError("Target data dictionary from " +
                        args.target_data + "is missing a key: " + str(e))
        elif isinstance(target_data, list):
            target_names = target_data
            class_weight = None
        else:
            raise ValueError("Target data must be list or dict, not " +
                    str(type(target_data)))

        n_classes = len(target_names)
    else:
        target_names = None
        class_weight = None
        if args.n_classes > -1:
            n_classes = args.n_classes
        else:
            n_classes = max(y_train)+1

    if class_weight is not None:
        # Keys are strings in JSON; convert them to int.
        for k,v in class_weight.iteritems():
            del class_weight[k]
            class_weight[int(k)] = v

    logging.debug("n_classes {0} min {1} max {2}".format(
        n_classes, min(y_train), max(y_train)))

    y_train_one_hot = np_utils.to_categorical(y_train, n_classes)
    y_validation_one_hot = np_utils.to_categorical(y_validation, n_classes)

    logging.debug("y_train_one_hot " + str(y_train_one_hot.shape))
    logging.debug("x_train " + str(x_train.shape))

    input_width = x_train.shape[1]

    min_vocab_index = np.min(x_train)
    max_vocab_index = np.max(x_train)
    logging.debug("min vocab index {0} max vocab index {1}".format(
        min_vocab_index, max_vocab_index))

    json_cfg = json.load(open(args.model_dir + '/model.json'))
    # Add a few values to the dictionary that are properties
    # of the training data.
    json_cfg['train_file'] = args.train_file
    json_cfg['validation_file'] = args.validation_file
    json_cfg['n_vocab'] = max(args.n_vocab, np.max(x_train) + 1)
    json_cfg['input_width'] = x_train.shape[1]
    json_cfg['n_classes'] = n_classes
    json_cfg['seed'] = args.seed

    logging.debug("loading model")

    sys.path.append(args.model_dir)
    from model import build_model
    model_cfg = ModelConfig(**json_cfg)
    model = build_model(model_cfg)

    setattr(model, 'stop_training', False)

    logging.info('model has {n_params} parameters'.format(
        n_params=count_parameters(model)))

    if args.extra_train_file is not None:
        callbacks = keras.callbacks.CallbackList()
    else:
        callbacks = []

    if not args.no_save:
        if args.description:
            with open(model_path + '/README.txt', 'w') as f:
                f.write(args.description + '\n')

        # Save model hyperparameters and code.
        for model_file in ['model.py', 'model.json']:
            shutil.copyfile(args.model_dir + '/' + model_file,
                    model_path + '/' + model_file)

        json.dump(vars(args), open(model_path + '/args.json', 'w'))

        # And weights.
        callbacks.append(ModelCheckpoint(
            filepath=model_path + '/model.h5',
            verbose=1,
            save_best_only=True))

    callback_logger = logging.info if args.log else callable_print
    callbacks.append(OptimizerMonitor(callback_logger))

    callbacks.append(EarlyStopping(
        monitor='val_loss', patience=model_cfg.patience, verbose=1))

    if args.classification_report:
        cr = ClassificationReport(x_validation, y_validation,
                callback_logger,
                target_names=target_names,
                error_classes_only=args.error_classes_only)
        callbacks.append(cr)

    if args.extra_train_file is not None:

        args.extra_train_file.append(args.train_file)
        logging.info("Using the following files for training: " +
                ','.join(args.extra_train_file))

        train_file_iter = itertools.cycle(args.extra_train_file)
        current_train = args.train_file

        callbacks._set_model(model)
        callbacks.on_train_begin(logs={})

        epoch = batch = 0

        while True:
            iteration = batch % len(args.extra_train_file)

            logging.info("epoch {epoch} iteration {iteration} - training with {train_file}".format(
                    epoch=epoch, iteration=iteration, train_file=current_train))
            callbacks.on_epoch_begin(epoch, logs={})

            n_train = x_train.shape[0]

            callbacks.on_batch_begin(batch, logs={'size': n_train})

            index_array = np.arange(n_train)
            if args.shuffle:
                index_array = rng.shuffle(index_array)

            batches = keras.models.make_batches(n_train, model_cfg.batch_size)
            logging.info("epoch {epoch} iteration {iteration} - starting {n_batches} batches".format(
                    epoch=epoch, iteration=iteration, n_batches=len(batches)))

            avg_train_loss = avg_train_accuracy = 0.
            for batch_index, (batch_start, batch_end) in enumerate(batches):
                batch_ids = index_array[batch_start:batch_end]

                train_loss, train_accuracy = model.train_on_batch(
                        x_train[batch_ids], y_train_one_hot[batch_ids],
                        accuracy=True, class_weight=class_weight)

                batch_end_logs = {'loss': train_loss, 'accuracy': train_accuracy}

                avg_train_loss = (avg_train_loss * batch_index + train_loss)/(batch_index + 1)
                avg_train_accuracy = (avg_train_accuracy * batch_index + train_accuracy)/(batch_index + 1)

                callbacks.on_batch_end(batch,
                        logs={'loss': train_loss, 'accuracy': train_accuracy})

            logging.info("epoch {epoch} iteration {iteration} - finished {n_batches} batches".format(
                    epoch=epoch, iteration=iteration, n_batches=len(batches)))

            logging.info("epoch {epoch} iteration {iteration} - loss: {loss} - acc: {acc}".format(
                    epoch=epoch, iteration=iteration, loss=avg_train_loss, acc=avg_train_accuracy))

            val_loss, val_acc = model.evaluate(
                    x_validation, y_validation_one_hot,
                    show_accuracy=True,
                    verbose=0 if args.log else 1)

            logging.info("epoch {epoch} iteration {iteration} - val_loss: {val_loss} - val_acc: {val_acc}".format(
                    epoch=epoch, iteration=iteration, val_loss=val_loss, val_acc=val_acc))
            epoch_end_logs = {'iteration': iteration, 'val_loss': val_loss, 'val_acc': val_acc}
            callbacks.on_epoch_end(epoch, epoch_end_logs)

            if model.stop_training:
                logging.info("epoch {epoch} iteration {iteration} - done training".format(
                    epoch=epoch, iteration=iteration))
                break

            current_train = next(train_file_iter)
            x_train, y_train = load_data(current_train,
                    args.data_name, args.target_name)
            y_train_one_hot = np_utils.to_categorical(y_train, n_classes)

            batch += 1
            if batch % len(args.extra_train_file) == 0:
                epoch += 1

            if epoch > n_epochs:
                break

        callbacks.on_train_end(logs={})
    else:
        model.fit(x_train, y_train_one_hot,
            shuffle=args.shuffle,
            nb_epoch=args.n_epochs,
            batch_size=model_cfg.batch_size,
            show_accuracy=True,
            validation_data=(x_validation, y_validation_one_hot),
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=2 if args.log else 1)

if __name__ == '__main__':
    sys.exit(main(get_parser()))
