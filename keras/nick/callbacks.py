import numpy as np
from keras.callbacks import Callback

class SklearnMetricCheckpointClassification(Callback):
    def __init__(self, model_path, x, y, metric='f1_score', verbose=0, save_best_only=False):
        super(Callback, self).__init__()
        self.__dict__.update(locals())
        self.best = np.Inf

    def on_epoch_end(self, epoch, logs={}):
        if self.save_best_only:
            self.model.predict_classses

            current = logs.get(self.monitor)
            if current is None:
                warnings.warn("Can save best model only with %s available, skipping." % (self.monitor), RuntimeWarning)
            else:
                if current < self.best:
                    if self.verbose > 0:
                        print("Epoch %05d: %s improved from %0.5f to %0.5f, saving model to %s"
                              % (epoch, self.monitor, self.best, current, self.filepath))
                    self.best = current
                    self.model.save_weights(self.filepath, overwrite=True)
                else:
                    if self.verbose > 0:
                        print("Epoch %05d: %s did not improve" % (epoch, self.monitor))
        else:
            if self.verbose > 0:
                print("Epoch %05d: saving model to %s" % (epoch, self.filepath))
            self.model.save_weights(self.filepath, overwrite=True)
