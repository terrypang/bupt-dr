import csv
import os
from datetime import datetime
from glob import glob
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import config
from utils import util


class TrainSplit(object):
    def __init__(self, eval_size=0.1, objective='classification'):
        self.eval_size = eval_size
        self.objective = objective

    def __call__(self, X, y, net):
        X_train = glob('{}/*/*'.format('data/train_' + str(config.crop_size)))
        X_train = shuffle(np.array(X_train), random_state=9)
        X_valid = glob('{}/*/*'.format('data/validation_' + str(config.crop_size)))
        X_valid = shuffle(np.array(X_valid), random_state=9)

        train_name = [os.path.basename(x).split('.')[0] for x in X_train]
        valid_name = [os.path.basename(x).split('.')[0] for x in X_valid]

        y_train = pd.read_csv(config.label_file, index_col=0).loc[train_name].values.flatten()
        y_valid = pd.read_csv(config.label_file, index_col=0).loc[valid_name].values.flatten()

        if self.objective == 'classification':
            y_train = np.array(y_train, dtype=np.int32)
            y_valid = np.array(y_valid, dtype=np.int32)
        elif self.objective == 'regression':
            y_train = np.array(y_train, dtype=np.float32)
            y_valid = np.array(y_valid, dtype=np.float32)
            y_train = np.vstack([l for l in y_train])
            y_valid = np.vstack([l for l in y_valid])

        return X_train, X_valid, y_train, y_valid


class Schedule(object):
    def __init__(self, name, schedule, weights_file=None):
        self.name = name
        self.schedule = schedule
        self.weights_file = weights_file

    def __call__(self, nn, train_history):
        epoch = train_history[-1]['epoch']
        if epoch in self.schedule:
            new_value = self.schedule[epoch]
            if new_value == 'stop':
                if self.weights_file is not None:
                    format_args = {
                        'timestamp': datetime.now().strftime('%Y-%m-%d-%H-%M-%S'),
                    }
                    weights_file = self.weights_file.format(**format_args)
                    nn.save_params_to(weights_file)
                raise StopIteration
            getattr(nn, self.name).set_value(np.cast['float32'](new_value))


class StepDecay(object):
    def __init__(self, name, start=0.03, stop=0.001, delay=0):
        self.name = name
        self.delay = delay
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, net, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop,
                                  net.max_epochs - self.delay)

        epoch = train_history[-1]['epoch'] - self.delay
        if epoch >= 0:
            new_value = np.cast['float32'](self.ls[epoch - 1])
            getattr(net, self.name).set_value(new_value)


class EarlyStopping(object):
    def __init__(self, patience=50, weights_file=None):
        self.patience = patience
        self.best_valid = np.inf
        self.best_valid_epoch = 0
        self.weights_file = weights_file

    def __call__(self, nn, train_history):
        current_valid = train_history[-1]['valid_loss']
        current_epoch = train_history[-1]['epoch']

        if not np.isnan(current_valid):
            if current_valid < self.best_valid:
                self.best_valid = current_valid
                self.best_valid_epoch = current_epoch

            if current_valid >= self.best_valid:
                if self.best_valid_epoch + self.patience < current_epoch:
                    print('Early stopping.')
                    print('Best valid loss was {:.6f} at epoch {}.'.format(self.best_valid, self.best_valid_epoch))
                    if self.weights_file is not None:
                        format_args = {
                            'timestamp': datetime.now().strftime('%Y-%m-%d-%H-%M-%S'),
                        }
                        weights_file = self.weights_file.format(**format_args)
                        nn.save_params_to(weights_file)
                    raise StopIteration()
                else:
                    return


class SaveBestWeights(object):
    def __init__(self, weights_file, loss='kappa', greater_is_better=True):
        self.weights_file = weights_file
        self.best_valid = np.inf
        self.loss = loss
        self.greater_is_better = greater_is_better
        self.queue = []

    def __call__(self, nn, train_history):
        current_valid = train_history[-1][self.loss] * (-1.0 if self.greater_is_better else 1.0)
        if current_valid < self.best_valid:
            format_args = {
                'loss': train_history[-1][self.loss],
                'timestamp': datetime.now().strftime('%Y-%m-%d-%H-%M-%S'),
                'epoch': '{:04d}'.format(train_history[-1]['epoch']),
            }
            weights_file = self.weights_file.format(**format_args)

            self.best_valid = current_valid
            nn.save_params_to(weights_file)
            self.queue.append(weights_file)
        if len(self.queue) > 5:
            file = self.queue.pop(0)
            os.remove(file)


class SaveLogs(object):
    def __init__(self, log_file):
        format_args = {
            'timestamp': datetime.now().strftime('%Y-%m-%d-%H-%M-%S'),
        }
        self.log_file = log_file.format(**format_args)
        with open(self.log_file, 'w') as f:
            fieldnames = ['timestamp', 'epoch', 'train_loss', 'valid_loss', 'acc', 'kappa']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    def __call__(self, nn, train_history):
        timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        epoch = train_history[-1]['epoch']
        train_loss = train_history[-1]['train_loss']
        valid_loss = train_history[-1]['valid_loss']
        acc = train_history[-1]['acc']
        kappa = train_history[-1]['kappa']
        with open(self.log_file, 'a') as f:
            fieldnames = ['timestamp', 'epoch', 'train_loss', 'valid_loss', 'acc', 'kappa']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow({'timestamp': timestamp, 'epoch': epoch, 'train_loss': train_loss,
                             'valid_loss': valid_loss, 'acc': acc, 'kappa': kappa})
