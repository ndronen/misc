import numpy as np
import h5py

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

def load_model_data(path, data_name, target_name):
    hdf5 = h5py.File(path)
    datasets = [hdf5[d].value.astype(np.int32) for d in data_name]
    for i,d in enumerate(datasets):
        if d.ndim == 1:
            datasets[i] = d.reshape((d.shape[0], 1))
    print([d.shape for d in datasets])
    data = np.concatenate(datasets, axis=1)
    target = hdf5[target_name].value.astype(np.int32)
    return data, target
