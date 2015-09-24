import os
import numpy as np
from keras.callbacks import Callback
import keras.callbacks
import numpy as np
import six
from sklearn.metrics import classification_report, fbeta_score

'''
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
'''

class ClassificationReport(Callback):
    def __init__(self, x, y, logger, target_names=None, error_classes_only=True, iteration_freq=10):
        self.x = x
        self.y = y
        self.logger = logger
        self.iteration_freq = iteration_freq

        if target_names is not None:
            if error_classes_only:
                labels, target_names = self.error_classes(target_names)
            else:
                labels = np.arange(len(target_names))
        else:
            labels = None

        self.labels = labels
        self.target_names = target_names

    def on_epoch_end(self, epoch, logs={}):
        if 'iteration' in logs.keys() and logs['iteration'] % self.iteration_freq != 0:
            # If we've broken a large training set into smaller chunks, we don't
            # need to run the classification report after every chunk.
            return

        y_hat = self.model.predict_classes(self.x, verbose=0)
        fbeta = fbeta_score(self.y, y_hat, beta=0.5, average='weighted')
        report = classification_report(
                self.y, y_hat,
                labels=self.labels, target_names=self.target_names)

        if 'iteration' in logs.keys():
            self.logger("epoch {epoch} iteration {iteration} - val_fbeta(beta=0.5): {fbeta}".format(
                epoch=epoch, iteration=logs['iteration'], fbeta=fbeta))
        else:
            self.logger("epoch {epoch} - val_fbeta(beta=0.5): {fbeta}".format(
                epoch=epoch, fbeta=fbeta))

        self.logger(report)

    def error_classes(self, target_names):
        # Assumes actual labels (numeric codes) start at 0 and are
        # contiguous.
        labels = np.arange(len(target_names))
        pairs = [pair.split('-') for pair in target_names]
        mask = np.array([pair[0] != pair[1] for pair in pairs])
        return labels[mask], target_names[mask]

class OptimizerMonitor(Callback):
    def __init__(self, logger):
        self.logger = logger

    def on_epoch_end(self, epoch, logs={}):
        if not hasattr(self.model.optimizer, 'lr'):
            return

        lr = self.model.optimizer.lr.get_value()
        optimizer_state = str({ 'lr': lr })

        if 'iteration' in logs.keys():
            self.logger("epoch {epoch} iteration {iteration} - optimizer state {optimizer_state}".format(
                epoch=epoch, iteration=logs['iteration'], optimizer_state=optimizer_state))
        else:
            self.logger("epoch {epoch} - optimizer state {optimizer_state}".format(
                epoch=epoch, optimizer_state=optimizer_state))

class VersionedModelCheckpoint(keras.callbacks.ModelCheckpoint):
    def __init__(self, filepath, monitor='val_loss', verbose=0, max_epochs=10000):
        save_best_only = False
        super(VersionedModelCheckpoint).__init__(
                filepath, monitor, verbose, save_best_only)
        self.basepath, self.ext = os.path.splitext(self.filepath)
        self.epoch = 0
        width = int(np.log10(max_epochs)) + 1
        self.fmt_string = '{basepath}-{epoch:0' + str(width) + 'd}.{ext}'

    def on_epoch_end(self, epoch, logs={}):
        if os.path.exists(self.filepath):
            newpath = self.fmt_string.format(
                    basepath=self.basepath, epoch=self.epoch, ext=self.ext)
            os.rename(self.filepath, newpath)
        self.epoch += 1
