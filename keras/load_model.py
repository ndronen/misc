# Run `run train-model.py` inside ipython before using this code.

import sys
import json
import numpy as np
from nick.utils import ModelConfig, load_model_data
from keras.utils.np_utils import to_categorical

def load_model(model_dir, input_file_arg='validation_file'):
    args_json = json.load(open(model_dir + '/args.json'))
    model_json = json.load(open(model_dir + '/model.json'))
    model_json.update(args_json)

    for k,v in model_json['model_cfg']:
        model_json[k] = v

    data, target = load_model_data(
            model_json[input_file_arg],
            model_json['data_name'], model_json['target_name'])

    model_json['input_width'] = data.shape[1]
    model_json['n_classes'] = np.max(target) + 1

    target_one_hot = to_categorical(target, model_json['n_classes'])

    # Re-instantiate ModelConfig using the updated JSON.
    sys.path.append(model_dir)
    from model import build_model
    model_cfg = ModelConfig(**model_json)
    model = build_model(model_cfg)

    # Load the saved weights.
    model.load_weights(model_dir + '/model.h5')

    return model, data, target_one_hot, target
