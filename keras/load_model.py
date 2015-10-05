# Run `run train-model.py` inside ipython before using this code.

import sys
import h5py
import json
import numpy as np
from nick.utils import ModelConfig, load_model_data
from keras.utils.np_utils import to_categorical
from sklearn import metrics

def load_model(model_dir, input_file_arg='validation_file'):
    args_json = json.load(open(model_dir + '/args.json'))
    print('args.json', args_json)
    model_json = json.load(open(model_dir + '/model.json'))
    print('model.json', model_json)
    model_json.update(args_json)
    print(model_json)

    if 'model_cfg' in model_json:
        for k,v in model_json['model_cfg']:
            model_json[k] = v

    data, target = load_model_data(
            model_json[input_file_arg],
            model_json['data_name'], model_json['target_name'])

    _, current_word_target = load_model_data(
            model_json[input_file_arg],
            model_json['data_name'], 'current_word_code')

    _, position = load_model_data(
            model_json[input_file_arg],
            model_json['data_name'], 'position')

    f = h5py.File(model_json[input_file_arg])
    length = f['len'].value
    f.close()

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
    model.load_weights(model_dir + '/model.h5')

    # Convert to a namespace for easy tab completion.
    results = {
            "model": model,
            "data": data,
            "target": target,
            "one_hot_target": one_hot_target,
            "current_word_target": current_word_target,
            "length": length,
            "position": position,
            "config": model_cfg
            }

    return ModelConfig(**results)
    #return model, data, one_hot_target, target, current_word_target, len

def print_classification_report(target, pred, target_names, digits=4):
    print(metrics.classification_report(target, pred, target_names=target_names, digits=digits))

def precision_recall_fscore_support(target, pred, beta=1.0, average='weighted'):
    prfs = metrics.precision_recall_fscore_support(target, pred, beta=beta, average=average)
    prfs = list(prfs)
    if average is not None:
        prfs[-1] = len(target)
    return prfs

def predict_with_absolute_threshold(probs, target, current_word_target, threshold=0.7):
    preds = np.argmax(probs, axis=1)
    preds_with_thresh = np.zeros_like(preds)

    for i in np.arange(0, len(preds)):
        pred = preds[i]
        prob = probs[i]
        if max(prob) < threshold:
            preds_with_thresh[i] = current_word_target[i]
        else:
            preds_with_thresh[i] = np.argmax(prob)

    return preds_with_thresh

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
