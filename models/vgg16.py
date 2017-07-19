from lasagne.layers.dnn import Pool2DDNNLayer
from utils.layers import conv2dbn
import lasagne as nn


def build_model(img_width, img_height, num_units):
    conv_kwargs = dict(
        pad='same',
        nonlinearity=nn.nonlinearities.very_leaky_rectify,
        W=nn.init.GlorotNormal(gain=1 / 3.0),
    )

    l = nn.layers.InputLayer(name='in', shape=(None, 3, img_width, img_height))

    # 512
    l = conv2dbn(l, name='l1c1', num_filters=32, filter_size=(7, 7), stride=2, **conv_kwargs)
    # l = nn.layers.dnn.MaxPool2DDNNLayer(l, name='l1p', pool_size=2)

    # 256
    l = conv2dbn(l, name='l2c1', num_filters=48, filter_size=(3, 3), stride=2, **conv_kwargs)
    l = conv2dbn(l, name='l2c2', num_filters=48, filter_size=(3, 3), **conv_kwargs)
    l = conv2dbn(l, name='l2c3', num_filters=48, filter_size=(3, 3), **conv_kwargs)

    # 128
    l = conv2dbn(l, name='l3c1', num_filters=64, filter_size=(3, 3), stride=2, **conv_kwargs)
    l = conv2dbn(l, name='l3c2', num_filters=64, filter_size=(3, 3), **conv_kwargs)
    l = conv2dbn(l, name='l3c3', num_filters=64, filter_size=(3, 3), **conv_kwargs)

    # 64
    l = conv2dbn(l, name='l4c1', num_filters=80, filter_size=(3, 3), stride=2, **conv_kwargs)
    l = conv2dbn(l, name='l4c2', num_filters=80, filter_size=(3, 3), **conv_kwargs)
    l = conv2dbn(l, name='l4c3', num_filters=80, filter_size=(3, 3), **conv_kwargs)
    l = conv2dbn(l, name='l4c4', num_filters=80, filter_size=(3, 3), **conv_kwargs)

    # 32
    l = conv2dbn(l, name='l5c1', num_filters=96, filter_size=(3, 3), stride=2, **conv_kwargs)
    l = conv2dbn(l, name='l5c2', num_filters=96, filter_size=(3, 3), **conv_kwargs)
    l = conv2dbn(l, name='l5c3', num_filters=96, filter_size=(3, 3), **conv_kwargs)
    l = conv2dbn(l, name='l5c4', num_filters=96, filter_size=(3, 3), **conv_kwargs)

    # 16
    l = conv2dbn(l, name='l6c1', num_filters=128, filter_size=(3, 3), stride=2, **conv_kwargs)
    l = conv2dbn(l, name='l6c2', num_filters=128, filter_size=(3, 3), **conv_kwargs)
    l = conv2dbn(l, name='l6c3', num_filters=128, filter_size=(3, 3), **conv_kwargs)
    l = conv2dbn(l, name='l6c4', num_filters=128, filter_size=(3, 3), **conv_kwargs)

    # 8
    l = Pool2DDNNLayer(l, name='gp', pool_size=8, mode='average_inc_pad')
    l = nn.layers.DropoutLayer(l, name='gpdrop', p=0.8)

    l = nn.layers.DenseLayer(l, name='out', num_units=num_units, nonlinearity=None)
    l = nn.layers.NonlinearityLayer(l, nonlinearity=nn.nonlinearities.softmax)

    return l