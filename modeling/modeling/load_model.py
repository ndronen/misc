# Run `run train-model.py` inside ipython before using this code.

import sys
import h5py
import json
import numpy as np
from modeling.utils import ModelConfig, load_model_data
from keras.utils.np_utils import to_categorical
from sklearn import metrics

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
