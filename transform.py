import click
import theano
import numpy as np
import config
from utils import util, iterator, tta, augmentation
from utils.nolearn_net import NeuralNet
import time
from glob import glob


@click.command()
@click.option('--n_iter', default=20, show_default=True,
              help="Iterations for test time averaging.")
@click.option('--skip', default=0, show_default=True,
              help="Number of test time averaging iterations to skip.")
def transform(n_iter, skip):
    file_path = 'data/convert_' + str(config.crop_size)
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

    weights_from = config.transform_weight

    args = {
        'layers': model.build_model(config.img_width, config.img_height, output_num, regression),
        'batch_iterator_train': iterator.ResampleIterator(batch_size=config.batch_size_train),
        'batch_iterator_test': iterator.SharedIterator(deterministic=True, batch_size=config.batch_size_test),
        'regression': regression,
        'update_learning_rate': theano.shared(np.cast['float32'](config.schedule[0])),
    }
    net = NeuralNet(**args)

    print("Model is {}, and objective is {}".format(config.model, config.objective))
    try:
        net.load_params_from(weights_from)
        print("loaded weights from {}".format(weights_from))
    except IOError:
        print("couldn't load weights starting from scratch")
        raise StopIteration()

    tfs, color_vecs = tta.build_quasirandom_transforms(n_iter, skip=skip, color_sigma=config.sigma, **config.aug_params)

    print("extracting features for files in {}".format(file_path))
    fs = sorted(glob('{}/*'.format(file_path)))
    Xs, Xs2 = None, None

    tic = time.time()
    for i, (tf, color_vec) in enumerate(zip(tfs, color_vecs), start=1):
        print("transform iter {}".format(i))
        X = net.transform(fs, config.output_layer, transform=tf, color_vec=color_vec)
        if Xs is None:
            Xs = X
            Xs2 = X ** 2
        else:
            Xs += X
            Xs2 += X ** 2

        print('took {:6.1f} seconds'.format(time.time() - tic))
        tic = time.time()
        if i % 5 == 0 or n_iter < 5:
            std = np.sqrt((Xs2 - Xs ** 2 / i) / (i - 1))
            util.save_mean(Xs / i, i, skip=skip)
            util.save_std(std, i, skip=skip)
            print('saved {} iterations'.format(i))


if __name__ == '__main__':
    transform()
