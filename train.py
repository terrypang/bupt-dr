import os
from glob import glob
import click
import numpy as np
import pandas as pd
import theano
from lasagne.updates import nesterov_momentum, rmsprop
from lasagne.objectives import categorical_crossentropy, squared_error
import config
from utils import event, iterator, util, losses, metrics
from utils.nolearn_net import NeuralNet
import theano.tensor as T


@click.command()
@click.option('--objective', default='regression', show_default=True,
              help='Objective type.')
@click.option('--weights_from', default=None, show_default=True,
              help='Path to initial weights file.')
def main(objective, weights_from):
    model = util.load_module(config.model)

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

    fs = glob('{}/*'.format('data/convert_' + str(config.crop_size)))
    names = [os.path.basename(x).split('.')[0] for x in fs]
    labels = pd.read_csv(config.label_file, index_col=0).loc[names].values.flatten()

    args = {
        'layers': model.build_model(config.img_width, config.img_height, output_num, regression),
        'batch_iterator_train': iterator.ResampleIterator(batch_size=config.batch_size_train),
        'batch_iterator_test': iterator.SharedIterator(deterministic=True, batch_size=config.batch_size_test),
        'on_epoch_finished': [
            event.Schedule('update_learning_rate', config.schedule, weights_file=util.final_weights_file()),
            # event.StepDecay('update_learning_rate', start=1e-2, stop=1e-5),
            event.SaveBestWeights(weights_file=util.weights_kappa_best(),
                            loss='kappa', greater_is_better=True, ),
            event.SaveBestWeights(weights_file=util.weights_loss_best(),
                                  loss='valid_loss', greater_is_better=False, ),
            event.SaveLogs(util.log_file()),
            event.EarlyStopping(weights_file=util.final_weights_file()),
        ],
        'train_split': event.TrainSplit(objective=objective),
        'objective_loss_function': objective_loss_function,
        'objective_l2': 0.0005,
        'y_tensor_type': y_tensor_type,
        'regression': regression,
        'max_epochs': 1000,
        'verbose': 2,
        'update_learning_rate': theano.shared(np.cast['float32'](config.schedule[0])),
        'update': nesterov_momentum,
        'update_momentum': 0.9,
        # 'update': rmsprop,
        # 'update_rho': 0.9,
        # 'update_epsilon': 1e-6,
        'custom_scores': [('acc', metrics.accuracy), ('kappa', metrics.kappa)],
    }
    net = NeuralNet(**args)

    print("Model is {}, and objective is {}".format(config.model, objective))
    try:
        net.load_params_from(weights_from)
        print("loaded weights from {}".format(weights_from))
    except IOError:
        print("couldn't load weights starting from scratch")

    print("fitting ...")
    net.fit(names, labels)


if __name__ == '__main__':
    main()
