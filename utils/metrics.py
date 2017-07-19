from utils.quadratic_weighted_kappa import quadratic_weighted_kappa
import numpy as np


def accuracy(y_true, y_pred):
    min_rating = 0
    max_rating = 4

    if y_pred.shape[1] == 5:
        y_pred = y_pred.argmax(axis=1)
    elif y_pred.shape[1] == 1:
        y_true = np.round(y_true).astype(int).ravel()
        y_pred[np.isnan(y_pred)] = 0
        y_pred = np.clip(y_pred, min_rating, max_rating)
        y_pred = np.round(y_pred).astype(int).ravel()
    else:
        raise TypeError('Prediction shape error')

    return sum([1.0 for x, y in zip(y_true, y_pred) if x == y]) / len(y_true)


def kappa(y_true, y_pred):
    min_rating = 0
    max_rating = 4

    if y_pred.shape[1] == 5:
        y_pred = y_pred.argmax(axis=1)
    elif y_pred.shape[1] == 1:
        y_true = np.round(y_true).astype(int).ravel()
        y_pred[np.isnan(y_pred)] = 0
        y_pred = np.clip(y_pred, min_rating, max_rating)
        y_pred = np.round(y_pred).astype(int).ravel()
    else:
        raise TypeError('Prediction shape error')

    return quadratic_weighted_kappa(y_true, y_pred, min_rating, max_rating)
