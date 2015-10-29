import sys
import os.path
import logging
import uuid
import json    
import numpy as np
import h5py

from keras.utils.np_utils import to_categorical
from sklearn import metrics

def count_parameters(model):
    if hasattr(model, 'count_params'):
        return model.count_params()
    else:
        n = 0
        for layer in model.layers:
            for param in layer.params:
                n += np.prod(param.shape.eval())
        return n

class ModelConfig:
    def __init__(self, **entries): 
        self.__dict__.update(entries)

    def __repr__(self):
        return str(vars(self.dict))

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

def callable_print(s):
    print(s)

def load_model_data(path, data_name, target_name, n=sys.maxint):
    hdf5 = h5py.File(path)
    datasets = [hdf5[d].value.astype(np.int32) for d in data_name]
    for i,d in enumerate(datasets):
        if d.ndim == 1:
            datasets[i] = d.reshape((d.shape[0], 1))
    print([d.shape for d in datasets])
    data = np.concatenate(datasets, axis=1)
    target = hdf5[target_name].value.astype(np.int32)

    if len(data) > n:
        target = target[0:n]
        data = data[0:n, :]

    return data, target

def setup_logging(args):
    if args.log and not args.no_save:
        logging.basicConfig(filename=args.model_path + 'model.log',
                format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                datefmt='%m-%d %H:%M',
                level=logging.DEBUG)
        stdout = LoggerWriter(logging.info)
        stderr = LoggerWriter(logging.warning)
    else:
        logging.basicConfig(level=logging.DEBUG)
        stdout = sys.stdout
        stderr = sys.stderr
    return stdout, stderr

def build_model_id(args):
    if args.model_dest:
        return args.model_dest
    else:
        return uuid.uuid1().hex

def build_model_path(args, model_id):
    # TODO: Windows compatibility.
    return args.model_dir + '/' + model_id + '/'

def setup_model_dir(args, model_path):
    if not args.no_save:
        if not os.path.exists(model_path):
            os.mkdir(model_path)
            print('created model directory ' + model_path)
        else:
            print('using existing model directory ' + model_path)

def load_model_json(args, x_train, n_classes):
    # Load the base model configuration.
    json_cfg = json.load(open(args.model_dir + '/model.json'))

    # Copy command-line arguments.
    for k,v in vars(args).iteritems():
        json_cfg[k] = v
    # Copy (overriding) model parameters provided on the command-line.
    for k,v in args.model_cfg:
        json_cfg[k] = v

    # Add some values are derived from the training data.
    json_cfg['n_vocab'] = max(args.n_vocab, np.max(x_train) + 1)
    json_cfg['input_width'] = x_train.shape[1]
    json_cfg['n_classes'] = n_classes

    return json_cfg

def load_model(model_dir, input_file_arg='validation_file', input_file=None):
    args_json = json.load(open(model_dir + '/args.json'))
    model_json = json.load(open(model_dir + '/model.json'))
    model_json.update(args_json)

    if 'model_cfg' in model_json:
        for k,v in model_json['model_cfg']:
            model_json[k] = v

    if input_file is not None:
        input_file_path = input_file
    else:
        input_file_path = model_json[input_file_arg]
    f = h5py.File(input_file_path)
    datasets = [f[d].value.astype(np.int32) for d in model_json['data_name']]
    for i,d in enumerate(datasets):
        if d.ndim == 1:
            datasets[i] = d.reshape((d.shape[0], 1))
    print([d.shape for d in datasets])
    data = np.concatenate(datasets, axis=1)
    target = f[model_json['target_name']].value.astype(np.int32)

    seen_keys = model_json['data_name']
    seen_keys.append(model_json['target_name'])

    model_json['input_width'] = data.shape[1]
    if 'n_classes' not in model_json:
        model_json['n_classes'] = np.max(target) + 1

    one_hot_target = to_categorical(target, model_json['n_classes'])

    # Re-instantiate ModelConfig using the updated JSON.
    sys.path.append(model_dir)
    from model import build_model
    model_cfg = ModelConfig(**model_json)
    model = build_model(model_cfg)

    # Load the saved weights.
    #model.load_weights(model_dir + '/model.h5')

    results = {
            "model": model,
            "data": data,
            "target": target,
            "one_hot_target": one_hot_target,
            "config": model_cfg
            }

    for key in f.keys():
        if key not in seen_keys:
            results[key] = f[key].value

    f.close()

    # Convert to a namespace for easy tab completion.
    return ModelConfig(**results)

def print_classification_report(target, pred, target_names, digits=4):
    print(metrics.classification_report(target, pred, target_names=target_names, digits=digits))

def precision_recall_fscore_support(target, pred, beta=1.0, average='weighted'):
    prfs = metrics.precision_recall_fscore_support(target, pred, beta=beta, average=average)
    prfs = list(prfs)
    if average is not None:
        prfs[-1] = len(target)
    return prfs

def predict_with_absolute_threshold(probs, target, threshold=0.7):
    preds = np.argmax(probs, axis=1)
    preds_with_thresh = []
    indices_used = []

    for i in np.arange(0, len(preds)):
        pred = preds[i]
        prob = probs[i]
        if max(prob) >= threshold:
            preds_with_thresh.append(np.argmax(prob))
            indices_used.append(i)

    return np.array(preds_with_thresh), np.array(indices_used)

def predict_with_min_margin_top_two(probs, target, current_word_target, margin=0.5):
    preds = np.argmax(probs, axis=1)
    preds_with_margin = np.zeros_like(target)

    for i in np.arange(0, len(preds)):
        pred = preds[i]
        prob = probs[i]
        next_most_prob, most_prob = prob[np.argsort(prob)[[-2,-1]]]

        if most_prob - next_most_prob < margin:
            preds_with_margin[i] = current_word_target[i]
        else:
            preds_with_margin[i] = np.argmax(prob)

    return preds_with_margin

def predict_with_min_margin_vs_actual(probs, target, current_word_target, min_margin=0.3):
    preds = np.argmax(probs, axis=1)
    preds_with_margin = np.zeros_like(preds)

    for i in np.arange(0, len(preds)):
        pred = preds[i]
        prob = probs[i]
        most_prob = prob[np.argsort(prob)][-1]
        actual_prob = prob[current_word_target[i]]
        margin = most_prob - actual_prob 

        if margin <= min_margin:
            preds_with_margin[i] = current_word_target[i]
        else:
            preds_with_margin[i] = np.argmax(prob)

    return preds_with_margin

def predict_with_minmax_margin(probs, target, current_word_target, min_margin=0.025, max_margin=0.25):
    preds = np.argmax(probs, axis=1)
    preds_with_margin = np.zeros_like(preds)

    for i in np.arange(0, len(preds)):
        pred = preds[i]
        prob = probs[i]
        most_prob = prob[np.argsort(prob)][-1]
        actual_prob = prob[current_word_target[i]]
        margin = most_prob - actual_prob 

        if margin > min_margin and margin < max_margin:
            preds_with_margin[i] = current_word_target[i]
        else:
            preds_with_margin[i] = np.argmax(prob)

    return preds_with_margin
