from lasagne.layers.dnn import Pool2DDNNLayer
from utils.layers import residual_block, conv2dbn
import lasagne as nn
import sys
sys.setrecursionlimit(10000)


def build_model(img_width, img_height, num_units):
    conv_kwargs = dict(
        pad='same',
        nonlinearity=nn.nonlinearities.very_leaky_rectify,
    )

    l = nn.layers.InputLayer(name='in', shape=(None, 3, img_width, img_height))

    # 512
    l = conv2dbn(l, name='l1c1', num_filters=32, filter_size=(7, 7), stride=2, **conv_kwargs)

    # 256
    l = nn.layers.dnn.MaxPool2DDNNLayer(l, name='l1p', pool_size=2)

    # 128
    for i in range(3):
        l = residual_block(
            l, name='2c%s' % i,
            # bottleneck=False,
            num_filters=48, filter_size=(3, 3),
            num_layers=2,
            **conv_kwargs
        )

    # 64
    for i in range(3):
        actual_stride = 2 if i == 0 else 1
        l = residual_block(
            l, name='3c%s' % i,
            # bottleneck=True, bottleneck_factor=4,
            num_filters=64, filter_size=(3, 3), stride=actual_stride,
            num_layers=2,
            **conv_kwargs
        )

    # 32
    for i in range(3):
        actual_stride = 2 if i == 0 else 1
        l = residual_block(
            l, name='4c%s' % i,
            # bottleneck=True, bottleneck_factor=4,
            num_filters=80, filter_size=(3, 3), stride=actual_stride,
            num_layers=3,
            **conv_kwargs
        )

    # 16
    for i in range(4):
        actual_stride = 2 if i == 0 else 1
        l = residual_block(
            l, name='5c%s' % i,
            # bottleneck=True, bottleneck_factor=4,
            num_filters=96, filter_size=(3, 3), stride=actual_stride,
            num_layers=3,
            **conv_kwargs
        )

    # 8
    # for i in range(5):
    #     actual_stride = 2 if i == 0 else 1
    #     l = residual_block(
    #         l, name='6c%s' % i,
    #         # bottleneck=True, bottleneck_factor=4,
    #         num_filters=128, filter_size=(3, 3), stride=actual_stride,
    #         num_layers=3,
    #         **conv_kwargs
    #     )

    # 8
    l = Pool2DDNNLayer(l, name='gp', pool_size=8, mode='average_inc_pad')
    l = nn.layers.DropoutLayer(l, name='gpdrop', p=0.5)

    l = nn.layers.DenseLayer(l, name='out', num_units=num_units, nonlinearity=None)
    l = nn.layers.NonlinearityLayer(l, nonlinearity=nn.nonlinearities.softmax)

    return l