from lasagne.layers.dnn import MaxPool2DDNNLayer
from utils.layers import conv2dbn, inceptionA, inceptionB, inceptionC, inceptionD, inceptionE
import lasagne as nn


def build_model(img_width, img_height, num_units):
    conv_kwargs = dict(
        nonlinearity=nn.nonlinearities.very_leaky_rectify,
        W=nn.init.GlorotNormal(gain=1 / 3.0),
    )

    l = nn.layers.InputLayer(name='in', shape=(None, 3, img_width, img_height))

    l = conv2dbn(l, name='1c1', num_filters=32, filter_size=3, stride=2, pad='same', **conv_kwargs)
    l = MaxPool2DDNNLayer(l, name='l1p', pool_size=2)
    l = conv2dbn(l, name='1c2', num_filters=32, filter_size=3, pad='same', **conv_kwargs)
    l = conv2dbn(l, name='1c3', num_filters=64, filter_size=3, pad=1, **conv_kwargs)
    l = MaxPool2DDNNLayer(l, name='1p', pool_size=3, stride=2)
    l = nn.layers.DropoutLayer(l, name='1cdrop', p=0.1)

    l = conv2dbn(l, name='2c1', num_filters=80, filter_size=1, pad='same', **conv_kwargs)
    l = conv2dbn(l, name='2c2', num_filters=192, filter_size=3, pad='same', **conv_kwargs)
    l = MaxPool2DDNNLayer(l, name='2p', pool_size=3, stride=2)
    l = nn.layers.DropoutLayer(l, name='2cdrop', p=0.1)

    l = inceptionA(
        l, name='3', nfilt=(
            (64,),
            (48, 64),
            (64, 96, 96),
            (32,)
        )
    )
    l = nn.layers.DropoutLayer(l, name='3cdrop', p=0.1)
    # l = inceptionA(
    #     l, name='4', nfilt=(
    #         (64,),
    #         (48, 64),
    #         (64, 96, 96),
    #         (64,)
    #     )
    # )
    # l = nn.layers.DropoutLayer(l, name='4cdrop', p=0.5)

    l = inceptionB(
        l, name='5', nfilt=(
            (384,),
            (64, 96, 96)
        )
    )
    l = nn.layers.DropoutLayer(l, name='5cdrop', p=0.1)

    # l = inceptionC(
    #     l, name='6', nfilt=(
    #         (192,),
    #         (128, 128, 192),
    #         (128, 128, 128, 128, 192),
    #         (192,)
    #     )
    # )
    # l = nn.layers.DropoutLayer(l, name='6cdrop', p=0.5)

    l = inceptionC(
        l, name='7', nfilt=(
            (192,),
            (192, 192, 192),
            (192, 192, 192, 192, 192),
            (192,)
        )
    )
    l = nn.layers.DropoutLayer(l, name='7cdrop', p=0.1)

    l = inceptionD(
        l, name='8', nfilt=(
            (192, 320),
            (192, 192, 192, 192)
        )
    )
    l = nn.layers.DropoutLayer(l, name='8cdrop', p=0.1)

    # l = inceptionE(
    #     l, name='9', nfilt=(
    #         (320,),
    #         (384, 384, 384),
    #         (448, 384, 384, 384),
    #         (192,)
    #     ),
    #     pool_type='avg'
    # )
    # l = nn.layers.DropoutLayer(l, name='9cdrop', p=0.5)

    # l = inceptionE(
    #     l, name='11', nfilt=(
    #         (320,),
    #         (384, 384, 384),
    #         (448, 384, 384, 384),
    #         (192,)
    #     ),
    #     pool_type='max'
    # )

    l = nn.layers.GlobalPoolLayer(l, name='gp')
    l = nn.layers.DropoutLayer(l, name='gpdrop', p=0.5)

    l = nn.layers.DenseLayer(l, name='out', num_units=num_units, nonlinearity=None)
    l = nn.layers.NonlinearityLayer(l, nonlinearity=nn.nonlinearities.softmax)

    return l