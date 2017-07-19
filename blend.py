import click
from glob import glob
import os
import config
import numpy as np
import pandas as pd
import theano
from lasagne import init
from lasagne.updates import adam, nesterov_momentum
from lasagne.nonlinearities import rectify
from lasagne.layers import DenseLayer, InputLayer, FeaturePoolLayer
from nolearn.lasagne import BatchIterator
from nolearn.lasagne import NeuralNet
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import theano.tensor as T
from lasagne.objectives import categorical_crossentropy, squared_error
import yaml
from utils import metrics, event, util, losses, iterator, nolearn_net
import predict_blend
import time

np.random.seed(9)

L1 = 2e-5
L2 = 0.005
N_ITER = 200
PATIENCE = 20
POWER = 0.5
N_HIDDEN_1 = 128
N_HIDDEN_2 = 32
BATCH_SIZE = 128

START_LR = 0.0005
SCHEDULE = {
    60: START_LR / 10.0,
    90: START_LR / 100.0,
    120: START_LR / 1000.0,
    150: START_LR / 5000.0,
    N_ITER: 'stop'
}


def per_patient_reshape(X, X_other=None):
    X_other = X if X_other is None else X_other
    # right_eye = np.arange(0, X.shape[0])[:, np.newaxis] % 2
    n = len(X)
    left_idx = np.arange(n)
    right_idx = left_idx + np.sign(2 * ((left_idx + 1) % 2) - 1)

    return np.hstack([X[left_idx], X_other[right_idx]]).astype(np.float32)


def load_features(fnames, test=False):
    if test:
        fnames = [os.path.join(os.path.dirname(f),
                               os.path.basename(f).replace('train', 'test'))
                  for f in fnames]

    data = [np.load(f) for f in fnames]
    data = [X.reshape([X.shape[0], -1]) for X in data]
    return np.hstack(data)


class TrainSplit(object):
    def __init__(self, tr, te):
        self.tr = tr
        self.te = te

    def __call__(self, X, y, net):
        if self.te is not None:
            print(len(self.tr))
            print(len(self.te))
            return X[self.tr], X[self.te], y[self.tr], y[self.te]
        else:
            print('Predict')
            return X, X[len(X):], y, y[len(y):]


class ResampleIterator(BatchIterator):

    def __init__(self, batch_size, resample_prob=0.2, shuffle_prob=0.5):
        self.resample_prob = resample_prob
        self.shuffle_prob = shuffle_prob
        super(ResampleIterator, self).__init__(batch_size)

    def __iter__(self):
        n_samples = self.X.shape[0]
        bs = self.batch_size
        indices = util.balance_per_class_indices(self.y.ravel())
        for i in range((n_samples + bs - 1) // bs):
            r = np.random.rand()
            if r < self.resample_prob:
                sl = indices[np.random.randint(0, n_samples, size=bs)]
            elif r < self.shuffle_prob:
                sl = np.random.randint(0, n_samples, size=bs)
            else:
                sl = slice(i * bs, (i + 1) * bs)
            Xb = self.X[sl]
            if self.y is not None:
                yb = self.y[sl]
            else:
                yb = None
            yield self.transform(Xb, yb)


def get_blend_net(objective, weights_from, tr, te):
    model = util.load_module(config.blend_model)

    if objective == 'classification':
        regression = False
        y_tensor_type = T.ivector
        objective_loss_function = categorical_crossentropy
        output_num = 5
    elif objective == 'regression':
        regression = True
        y_tensor_type = T.TensorType(theano.config.floatX, (False, False))
        objective_loss_function = squared_error
        output_num = 1
    elif objective == 'kappa':
        regression = True
        y_tensor_type = T.imatrix
        objective_loss_function = losses.quad_kappa_loss
        output_num = 5
    else:
        raise TypeError('objective type error')

    if weights_from is None:
        weights_from = util.weights_from()
    else:
        weights_from = str(weights_from)

    args = {
        'layers': model.build_model(config.blend_depth, config.blend_size, output_num, regression),
        'batch_iterator_train': ResampleIterator(batch_size=BATCH_SIZE),
        'batch_iterator_test': BatchIterator(batch_size=BATCH_SIZE),
        'on_epoch_finished': [
            event.Schedule('update_learning_rate', SCHEDULE, weights_file=util.final_weights_file()),
            event.SaveLogs(util.log_file()),
        ],
        'train_split': TrainSplit(tr, te),
        'objective_loss_function': objective_loss_function,
        'objective_l2': 0.0005,
        'y_tensor_type': y_tensor_type,
        'regression': regression,
        'max_epochs': 1000,
        'verbose': 2,
        'update_learning_rate': theano.shared(np.cast['float32'](START_LR)),
        'update': nesterov_momentum,
        'update_momentum': 0.9,
        # 'update': rmsprop,
        # 'update_rho': 0.9,
        # 'update_epsilon': 1e-6,
        'custom_scores': [('acc', metrics.accuracy), ('kappa', metrics.kappa)] if te is not None else None,
    }
    net = NeuralNet(**args)

    print("Model is {}, and objective is {}".format(config.blend_model, objective))
    try:
        net.load_params_from(weights_from)
        print("loaded weights from {}".format(weights_from))
    except IOError:
        print("couldn't load weights starting from scratch")

    return net


@click.command()
@click.option('--predict', is_flag=True, default=False, show_default=True,
              help="Make predictions on test set features after training.")
@click.option('--n_iter', default=1, show_default=True,
              help="Number of times to fit and average.")
@click.option('--objective', default='regression', show_default=True,
              help='Objective type.')
@click.option('--eval_size', default=0.1, show_default=True,
              help="Percentage of images to evaluate.")
@click.option('--weights_from', default=None, show_default=True,
              help='Path to initial weights file.')
def main(objective, weights_from, eval_size, n_iter, predict):
    file_path = 'data/convert_' + str(config.crop_size)
    fs = sorted(glob('{}/*'.format(file_path)))
    names = [os.path.basename(x).split('.')[0] for x in fs]
    labels = pd.read_csv(config.label_file, index_col=0).loc[names].values.flatten()
    labels = np.array(labels, dtype=np.float32)
    tr, te = util.split_indices(names, labels, test_size=eval_size)
    if predict:
        te = None

    print("fitting ...")

    runs = util.parse_blend_config(yaml.load(open('blend.yml')))
    scalers = {run: StandardScaler() for run in runs}
    y_preds = []
    for i in range(n_iter):
        print("iteration {} / {}".format(i + 1, n_iter))
        for run, files in runs.items():
            print("fitting features for run {}".format(run))
            print(files)
            X = load_features(files)
            X = scalers[run].fit_transform(X)
            print(X.shape)
            X = per_patient_reshape(X)
            print(X.shape)
            X = X.reshape(X.shape[0], config.blend_depth, config.blend_size, config.blend_size)
            print(X.shape)
            net = get_blend_net(objective, weights_from, tr, te)
            net.fit(X, labels)

            if not predict:
                y_pred = net.predict(X[te]).ravel()
                y_preds.append(y_pred)
                y_pred[np.isnan(y_pred)] = 0
                y_pred = np.clip(y_pred, 0, 4)
                y_pred = np.round(y_pred).astype(int).ravel()
                y_pred = y_pred.reshape(-1, 1)
                print("accuracy after run {}, iter {}: {}".format(
                    run, i, metrics.accuracy(labels[te], y_pred)))
                print("kappa after run {}, iter {}: {}".format(
                    run, i, metrics.kappa(labels[te], y_pred)))
                print("confusion matrix")
                print(confusion_matrix(labels[te], y_pred))
            else:
                y_pred = net.predict(X).ravel()
                y_pred[np.isnan(y_pred)] = 0
                y_pred = np.clip(y_pred, 0, 4)
                y_pred = np.round(y_pred).astype(int).ravel()
                y_pred = y_pred.reshape(-1, 1)
                print("accuracy after run {}, iter {}: {}".format(
                    run, i, metrics.accuracy(labels, y_pred)))
                print("kappa after run {}, iter {}: {}".format(
                    run, i, metrics.kappa(labels, y_pred)))
                print("confusion matrix")
                print(confusion_matrix(labels, y_pred))

    if not predict:
        final_pred = np.mean(y_preds, axis=0)
        final_pred[np.isnan(final_pred)] = 0
        final_pred = np.clip(final_pred, 0, 4)
        final_pred = np.round(final_pred).astype(int).ravel()
        final_pred = final_pred.reshape(-1, 1)
        print("Blend final kappa score: {}".format(metrics.kappa(labels[te], final_pred)))
        print("confusion matrix")
        print(confusion_matrix(labels[te], final_pred))


def testAll():
    objective = 'regression'
    weights_from = [
                    # 'weights/drnet/weights_final_regression_blend_regression_features_50_0.pkl',
                    # 'weights/drnet/weights_final_regression_blend_regression_features_50_50.pkl',
                    # 'weights/drnet/weights_final_regression_blend_regression_features_50_100.pkl',
                    'weights/drnet/weights_final_regression_blend_classification_features_50_0.pkl',
                    'weights/drnet/weights_final_regression_blend_classification_features_50_50.pkl',
                    'weights/drnet/weights_final_regression_blend_classification_features_50_100.pkl']
    file_path = 'data/convert_' + str(config.crop_size)
    fs = sorted(glob('{}/*'.format(file_path)))
    names = [os.path.basename(x).split('.')[0] for x in fs]
    labels = pd.read_csv(config.label_file, index_col=0).loc[names].values.flatten()
    labels = np.array(labels, dtype=np.float32)
    tr, te = None, None

    runs = util.parse_blend_config(yaml.load(open('blend.yml')))
    scalers = {run: StandardScaler() for run in runs}
    y_preds = []

    i = 0
    for run, files in runs.items():
        print("fitting features for run {}".format(run))
        X = load_features(files)
        X = scalers[run].fit_transform(X)
        print(X.shape)
        X = per_patient_reshape(X)
        print(X.shape)
        X = X.reshape(X.shape[0], config.blend_depth, config.blend_size, config.blend_size)
        print(X.shape)
        net = get_blend_net(objective, weights_from[i], tr, te)
        y_pred = net.predict(X).ravel()
        y_preds.append(y_pred)
        i = i+1

    final_pred = np.mean(y_preds, axis=0)
    final_pred[np.isnan(final_pred)] = 0
    final_pred = np.clip(final_pred, 0, 4)
    final_pred = np.round(final_pred).astype(int).ravel()
    final_pred = final_pred.reshape(-1, 1)
    print("Blend final accuracy score: {}".format(metrics.accuracy(labels, final_pred)))
    print("Blend final kappa score: {}".format(metrics.kappa(labels, final_pred)))
    print("confusion matrix")
    print(confusion_matrix(labels, final_pred))


def test():
    model = util.load_module(config.model)

    if config.objective == 'classification':
        regression = False
        output_num = 5
    elif config.objective == 'regression':
        regression = True
        output_num = 1
    elif config.objective == 'kappa':
        regression = True
        output_num = 5
    else:
        raise TypeError('objective type error')

    weights_from = config.predict_weight

    args = {
        'layers': model.build_model(config.img_width, config.img_height, output_num, regression),
        'batch_iterator_train': iterator.ResampleIterator(batch_size=config.batch_size_train),
        'batch_iterator_test': iterator.SharedIterator(deterministic=True, batch_size=config.batch_size_test),
        'regression': regression,
        'update_learning_rate': theano.shared(np.cast['float32'](config.schedule[0])),
    }
    feature_net = nolearn_net.NeuralNet(**args)

    print("Model is {}, and objective is {}".format(config.model, config.objective))
    try:
        feature_net.load_params_from(weights_from)
        print("loaded weights from {}".format(weights_from))
    except IOError:
        print("couldn't load weights starting from scratch")
        raise StopIteration()

    blend_nets = []
    for weight in config.blend_weights:
        net = predict_blend.get_blend_net(weight)
        blend_nets.append(net)

    file_path = 'data/test'
    fs = sorted(glob('{}/*'.format(file_path)))
    names = [os.path.basename(x).split('.')[0] for x in fs]
    labels = pd.read_csv(config.label_file, index_col=0).loc[names].values.flatten()
    labels = np.array(labels, dtype=np.float32)

    data = []
    result = []
    tic = time.time()
    data.append(predict_blend.transform(feature_net, fs, 20, 0))
    print('took {:6.1f} seconds'.format(time.time() - tic))
    data.append(predict_blend.transform(feature_net, fs, 20, 50))
    print('took {:6.1f} seconds'.format(time.time() - tic))
    data.append(predict_blend.transform(feature_net, fs, 20, 100))
    print('took {:6.1f} seconds'.format(time.time() - tic))

    i = 0
    for blend_net in blend_nets:
        print(data[i])
        print(data[i].shape)
        X = data[i].reshape(data[i].shape[0], config.blend_depth, config.blend_size, config.blend_size)
        y_pred = blend_net.predict(X)
        result.append(y_pred)
        i += 1;

    result = np.mean(result, axis=0)
    result[np.isnan(result)] = 0
    result = np.clip(result, 0, 4)
    result = np.round(result).astype(int).ravel()
    result = result.reshape(-1, 1)
    print("Blend final accuracy score: {}".format(metrics.accuracy(labels, result)))
    print("Blend final kappa score: {}".format(metrics.kappa(labels, result)))
    print("confusion matrix")
    print(confusion_matrix(labels, result))


if __name__ == '__main__':
    main()
    # testAll()
    # test()