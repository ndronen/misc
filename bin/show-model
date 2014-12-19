#!/usr/bin/env python

import sys
import joblib
import numpy as np
import matplotlib.pylab as plt

def show_model(model, plot=True, show_test=False):
    if 'train_y_misclass' in model.monitor.channels:
        train = model.monitor.channels['train_y_misclass'].val_record
        valid = model.monitor.channels['valid_y_misclass'].val_record
        test = model.monitor.channels['test_y_misclass'].val_record
    else:
        train = model.monitor.channels['train_y_mse'].val_record
        valid = model.monitor.channels['valid_y_mse'].val_record
        test = model.monitor.channels['test_y_mse'].val_record

    i = np.argmin(valid)

    if show_test:
        print("train " + str(train[i]) +
            " valid " + str(valid[i]) +
            " test " + str(test[i]))
    else:
        print("train " + str(train[i]) +
            " valid " + str(valid[i]))

    if plot:
        plt.plot(train, label='train')
        plt.plot(valid, label='valid')
        if show_test:
            plt.plot(test, label='test')
        plt.legend()
        plt.show(block=False)

if __name__ == '__main__':
    model_path = sys.argv[1]
    model = joblib.load(model_path)
    show_model(model, plot=False)
