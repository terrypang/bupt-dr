import importlib
import os
import numpy as np
import config
from sklearn import cross_validation
from utils.quadratic_weighted_kappa import quadratic_weighted_kappa


def load_module(mod):
    return importlib.import_module(mod.replace('/', '.').split('.py')[0])


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError:
        pass


def one_hot(vec):
    output_num = 5
    return np.eye(output_num)[vec].astype('int32')


def kappa(y_true, y_pred):
    min_rating = 0
    max_rating = 4
    y_true = y_true.flatten().astype(int)
    y_pred = y_pred.argmax(axis=1).astype(int)
    return quadratic_weighted_kappa(y_true, y_pred, min_rating, max_rating)


# Computes the classification accuracy
# https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/elementwise.py
def ce(y_true, y_pred):
    y_true = y_true.flatten().astype(int)
    y_pred = y_pred.argmax(axis=1).astype(int)
    return (sum([1.0 for x,y in zip(y_true,y_pred) if x == y]) /
            len(y_true))


def balance_per_class_indices(y, weights=config.balance['class_weights']):
    # y = np.argmax(y, axis=1)
    weights = np.array(weights, dtype=float)
    p = np.zeros(len(y))
    for i, weight in enumerate(weights):
        index = 0
        for j in range(len(y)):
            if y[index]==i:
                p[index] = weight
            index += 1
    return np.random.choice(np.arange(len(y)), size=len(y), replace=True,
                            p=np.array(p) / p.sum())


def split_indices(names, labels, test_size=0.1, random_state=None):
    left = np.array(['left' in n for n in names])
    left_right_labels = np.vstack([labels[left], labels[~left]]).T
    spl = cross_validation.StratifiedShuffleSplit(left_right_labels[:, 0],
                                                  test_size=test_size,
                                                  random_state=random_state,
                                                  n_iter=1)
    tr, te = next(iter(spl))
    tr = np.hstack([tr * 2, tr * 2 + 1])
    te = np.hstack([te * 2, te * 2 + 1])
    return tr, te


def weights_from():
    path = "weights/{}".format(config.model.split('/')[-1].split('.')[0])
    mkdir(path)
    return os.path.join(path, 'weights_fine_tune.pkl')


def final_weights_file():
    path = "weights/{}".format(config.model.split('/')[-1].split('.')[0])
    mkdir(path)
    return os.path.join(path, 'weights_final_{timestamp}.pkl')


def weights_loss_best():
    path = "weights/{}/loss_best".format(config.model.split('/')[-1].split('.')[0])
    mkdir(path)
    return os.path.join(path, '{epoch}_{timestamp}_{loss}.pkl')

def weights_kappa_best():
    path = "weights/{}/kappa_best".format(config.model.split('/')[-1].split('.')[0])
    mkdir(path)
    return os.path.join(path, '{epoch}_{timestamp}_{loss}.pkl')


def log_file():
    path = "logs/{}".format(config.model.split('/')[-1].split('.')[0])
    mkdir(path)
    return os.path.join(path, '{timestamp}.csv')


def save_mean(X, n_iter, skip=0):
    model_name = config.model.split('/')[-1].split('.')[0]
    fname = '{}_{}_mean_iter_{}_skip_{}.npy'.format(model_name, config.objective, n_iter, skip)
    np.save(open(os.path.join(config.feature_path, fname), 'wb'), X)


def save_std(X, n_iter, skip=0):
    model_name = config.model.split('/')[-1].split('.')[0]
    fname = '{}_{}_std_iter_{}_skip_{}.npy'.format(model_name, config.objective, n_iter, skip)
    np.save(open(os.path.join(config.feature_path, fname), 'wb'), X)


def getResult(y_pred, min_rating=0, max_rating=4):
    y_pred = np.clip(y_pred, min_rating, max_rating)
    y_pred = np.round(y_pred).astype(int).ravel()
    if y_pred == 0:
        return 'No DR'
    elif y_pred == 1:
        return 'Mild'
    elif y_pred ==2:
        return 'Moderate'
    elif y_pred == 3:
        return 'Severe'
    elif y_pred == 4:
        return 'Proliferative DR'


def parse_blend_config(cnf):
    return {run: [os.path.join(config.feature_path, f) for f in files]
            for run, files in cnf.items()}
