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
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from keras.utils import np_utils
from keras.optimizers import SGD
import keras.callbacks
from keras.callbacks import ModelCheckpoint, EarlyStopping
import keras.models
#from nick.callbacks import SklearnMetricCheckpoint

sys.path.append('.')

class ClassificationReport(keras.callbacks.Callback):
    def __init__(self, x, y, logger, target_names=None, msg='', error_classes_only=True):
        self.x = x
        self.y = y
        self.logger = logger
        self.msg = msg

        if target_names is not None:
            if error_classes_only:
                labels, target_names = self.error_classes(target_names)
            else:
                labels = np.arange(len(target_names))
        else:
            labels = None

        self.labels = labels
        self.target_names = target_names

    def on_epoch_end(self, batch, logs={}):
        y_hat = self.model.predict_classes(self.x, verbose=0)
        report = classification_report(
                self.y, y_hat,
                labels=self.labels, target_names=self.target_names)
        if self.msg and len(self.msg):
            self.logger(self.msg)
        self.logger(report)

    def error_classes(self, target_names):
        # Assumes actual labels (numeric codes) start at 0 and are
        # contiguous.
        labels = np.arange(len(target_names))
        pairs = [pair.split('-') for pair in target_names]
        mask = np.array([pair[0] != pair[1] for pair in pairs])
        return labels[mask], target_names[mask]

class ConfusionMatrix(keras.callbacks.Callback):
    def __init__(self, x, y, logger, msg=''):
        self.x = x
        self.y = y
        self.logger = logger
        self.msg = msg

    def on_epoch_end(self, batch, logs={}):
        y_hat = self.model.predict_classes(self.x, verbose=0)
        report = confusion_matrix(
                self.y, y_hat)
        if self.msg and len(self.msg):
            self.logger(self.msg)
        self.logger(report)

class ModelConfig:
    def __init__(self, **entries): 
        self.__dict__.update(entries)

    def __repr__(self):
        return '<%s>' % \
            str('\n '.join('%s : %s' % \
                (k, repr(v)) for (k, v) in self.__dict__.iteritems()))

class LoggerWriter:
    def __init__(self, level):
        # self.level is really like using log.debug(message)
        # at least in my case
        self.level = level

    def write(self, message):
        # if statement reduces the amount of newlines that are
        # printed to the logger
        if message != '\n':
            self.level(message)
                
    def flush(self):
        # create a flush method so things can be flushed when
        # the system wants to. Not sure if simply 'printing'
        # sys.stderr is the correct way to do it, but it seemed
        # to work properly for me.
        self.level(sys.stderr)

def fake_print(s):
    print(s)

def load_data(path, y_name, X_name='X'):
    data = h5py.File(path)
    x_data = data[X_name].value.astype(np.int32)
    y_data = data[y_name].value.astype(np.int32)
    return x_data, y_data

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
    parser.add_argument('target', metavar='TARGET_NAME', type=str,
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
    parser.add_argument('--n-train', default=np.inf, type=int,
            help='The number of training examples to use')
    parser.add_argument('--n-validation', default=np.inf, type=int,
            help='The number of validation examples to use')
    parser.add_argument('--n-vocab', default=-1, type=int,
            help='The number of words in the training vocabulary')
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
                level=logging.DEBUG)
        sys.stdout = LoggerWriter(logging.info)
        sys.stderr = LoggerWriter(logging.warning)
    else:
        logging.basicConfig(level=logging.DEBUG)

    x_train, y_train = load_data(args.train_file, args.target)
    x_validation, y_validation = load_data(
            args.validation_file, args.target)

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
    
    np.random.seed(args.seed)

    if args.target_data:
        target_names_dict = json.load(open(args.target_data))

        try:
            target_data = target_names_dict[args.target]
        except KeyError:
            raise ValueError("Invalid key " + args.target +
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

    logging.debug("loading model")

    sys.path.append(args.model_dir)
    from model import build_model
    model_cfg = ModelConfig(**json_cfg)
    model = build_model(model_cfg)

    callbacks = keras.callbacks.CallbackList()

    if not args.no_save:
        if args.description:
            with open(model_path + '/README.txt', 'w') as f:
                f.write(args.description + '\n')

        # Save model hyperparameters and code.
        for model_file in ['model.py', 'model.json']:
            shutil.copyfile(args.model_dir + '/' + model_file,
                    model_path + '/' + model_file)

        # And weights.
        callbacks.append(ModelCheckpoint(
            filepath=model_path + '/model.h5',
            verbose=1,
            save_best_only=True))

    callbacks.append(EarlyStopping(
        monitor='val_loss', patience=model_cfg.patience, verbose=1))

    callback_logger = logging.info if args.log else fake_print

    if args.classification_report:
        cr = ClassificationReport(x_validation, y_validation,
                callback_logger,
                msg='Validation set metrics',
                target_names=target_names,
                error_classes_only=args.error_classes_only)
        callbacks.append(cr)

    callbacks._set_model(model)

    if args.extra_train_file is not None:
        args.extra_train_file.append(args.train_file)
        logging.info("Using the following files for training: " +
                ','.join(args.extra_train_file))

        train_file_iter = itertools.cycle(args.extra_train_file)
        current_train = args.train_file
        epoch = batch = 0
        callbacks.on_train_begin(logs={})
        while True:
            logging.info("epoch {0} - training with {1}".format(
                    epoch, current_train))
            callbacks.on_epoch_begin(epoch, logs={})

            callbacks.on_batch_begin(batch,
                    logs={'size': x_train.shape[0]})

            batches = keras.models.make_batches(x_train.shape[0], model_cfg.batch_size)
            logging.debug("epoch {0} - starting {1} batches".format(
                    epoch, len(batches)))

            avg_train_loss = avg_train_accuracy = 0.
            for batch_index, (batch_start, batch_end) in enumerate(batches):
                train_loss, train_accuracy = model.train_on_batch(
                        x_train[batch_start:batch_end],
                        y_train_one_hot[batch_start:batch_end],
                        accuracy=True, class_weight=class_weight)

                batch_end_logs = {'loss': train_loss, 'accuracy': train_accuracy}

                avg_train_loss = (avg_train_loss * batch_index + train_loss)/(batch_index + 1)
                avg_train_accuracy = (avg_train_accuracy * batch_index + train_accuracy)/(batch_index + 1)

                callbacks.on_batch_end(batch,
                        logs={'loss': train_loss, 'accuracy': train_accuracy})

            logging.debug("epoch {0} - finished {1} batches".format(
                    epoch, len(batches)))
            logging.debug("epoch {0} - loss - {1} - accuracy {2}".format(
                    epoch, avg_train_loss, avg_train_accuracy))

            val_loss, val_accuracy = model.evaluate(x_validation,
                    y_validation_one_hot, show_accuracy=True,
                    verbose=2 if args.log else 1)

            epoch_end_logs = {'val_loss': val_loss, 'val_accuracy': val_accuracy}
            logging.info("epoch {0} - val_loss {1} - val_accuracy {2}".format(
                    epoch, val_loss, val_accuracy))
            callbacks.on_epoch_end(epoch, epoch_end_logs)

            if model.stop_training:
                logging.info("epoch {0} - done training".format(epoch))
                break

            current_train = next(train_file_iter)
            x_train, y_train = load_data(current_train, args.target)
            y_train_one_hot = np_utils.to_categorical(y_train, n_classes)

            epoch += 1
            batch += 1

        callbacks.on_train_end(logs={})
    else:
        model.fit(x_train, y_train_one_hot,
            shuffle=False,
            batch_size=model_cfg.batch_size,
            show_accuracy=True,
            validation_data=(x_validation, y_validation_one_hot),
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=2 if args.log else 1)

if __name__ == '__main__':
    sys.exit(main(get_parser()))
