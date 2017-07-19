from lasagne.layers.dnn import MaxPool2DDNNLayer, Conv2DDNNLayer
from utils.layers import RMSPoolLayer
import lasagne as nn


def build_model(img_width, img_height, output_num, regression):
    l = build_448model(img_width, img_height, output_num, regression)
    return l


def build_112model(img_width, img_height, output_num, regression):
    conv_kwargs = dict(
        pad='same',
        nonlinearity=nn.nonlinearities.leaky_rectify,
        W=nn.init.Orthogonal(gain=1.0),
        b=nn.init.Constant(0.05),
        untie_biases=True,
    )
    dense_kwargs = dict(
        nonlinearity=nn.nonlinearities.leaky_rectify,
        W=nn.init.Orthogonal(1.0),
        b=nn.init.Constant(0.05),
    )

    l = nn.layers.InputLayer(name='in', shape=(None, 3, img_width, img_height))
    # 3 x 112 x 112
    l = Conv2DDNNLayer(l, name='conv2d_1_5x5', num_filters=32, filter_size=(5, 5), stride=(2, 2), **conv_kwargs)
    # 32 x 56 x 56
    l = Conv2DDNNLayer(l, name='conv2d_2_3x3', num_filters=32, filter_size=(3, 3), **conv_kwargs)
    # 32 x 56 x 56
    l = MaxPool2DDNNLayer(l, name='maxpool_3_3x3', pool_size=3, stride=(2, 2))
    # 32 x 27 x 27
    l = Conv2DDNNLayer(l, name='conv2d_4_5x5', num_filters=64, filter_size=(5, 5), stride=(2, 2), **conv_kwargs)
    # 64 x 14 x 14
    l = Conv2DDNNLayer(l, name='conv2d_5_3x3', num_filters=64, filter_size=(3, 3), **conv_kwargs)
    # 64 x 14 x 14
    l = Conv2DDNNLayer(l, name='conv2d_6_3x3', num_filters=64, filter_size=(3, 3), **conv_kwargs)
    # 64 x 14 x 14
    l = MaxPool2DDNNLayer(l, name='maxpool_7_3x3', pool_size=3, stride=(2, 2))
    # 64 x 6 x 6
    l = Conv2DDNNLayer(l, name='conv2d_8_3x3', num_filters=128, filter_size=(3, 3), **conv_kwargs)
    # 128 x 6 x 6
    l = Conv2DDNNLayer(l, name='conv2d_9_3x3', num_filters=128, filter_size=(3, 3), **conv_kwargs)
    # 128 x 6 x 6
    l = Conv2DDNNLayer(l, name='conv2d_10_3x3', num_filters=128, filter_size=(3, 3), **conv_kwargs)
    # 128 x 6 x 6
    l = RMSPoolLayer(l, name='rmspool_18_3x3', pool_size=3, stride=(3, 3))
    # 128 x 2 x 2
    l = nn.layers.DropoutLayer(l, name='drop_19', p=0.5)
    # 128 x 2 x 2
    l = nn.layers.DenseLayer(l, name='dense_20', num_units=1024, **dense_kwargs)
    # 1024
    l = nn.layers.FeaturePoolLayer(l, name='featurepool_21', pool_size=2)
    # 512
    l = nn.layers.DropoutLayer(l, name='drop_22', p=0.5)
    # 512
    l = nn.layers.DenseLayer(l, name='dense_23', num_units=1024, **dense_kwargs)
    # 1024
    l = nn.layers.FeaturePoolLayer(l, name='featurepool_24', pool_size=2)
    # 512
    l = nn.layers.DenseLayer(l, name='out_25', num_units=output_num)

    return l


def build_224model(img_width, img_height, output_num, regression):
    conv_kwargs = dict(
        pad='same',
        nonlinearity=nn.nonlinearities.leaky_rectify,
        W=nn.init.Orthogonal(gain=1.0),
        b=nn.init.Constant(0.05),
        untie_biases=True,
    )
    dense_kwargs = dict(
        nonlinearity=nn.nonlinearities.leaky_rectify,
        W=nn.init.Orthogonal(1.0),
        b=nn.init.Constant(0.05),
    )

    l = nn.layers.InputLayer(name='in', shape=(None, 3, img_width, img_height))
    # 3 x 224 x 224
    l = Conv2DDNNLayer(l, name='conv2d_1_5x5', num_filters=32, filter_size=(5, 5), stride=(2, 2), **conv_kwargs)
    # 32 x 112 x 112
    l = Conv2DDNNLayer(l, name='conv2d_2_3x3', num_filters=32, filter_size=(3, 3), **conv_kwargs)
    # 32 x 112 x 112
    l = MaxPool2DDNNLayer(l, name='maxpool_3_3x3', pool_size=3, stride=(2, 2))
    # 32 x 55 x 55
    l = Conv2DDNNLayer(l, name='conv2d_4_5x5', num_filters=64, filter_size=(5, 5), stride=(2, 2), **conv_kwargs)
    # 64 x 28 x 28
    l = Conv2DDNNLayer(l, name='conv2d_5_3x3', num_filters=64, filter_size=(3, 3), **conv_kwargs)
    # 64 x 28 x 28
    l = Conv2DDNNLayer(l, name='conv2d_6_3x3', num_filters=64, filter_size=(3, 3), **conv_kwargs)
    # 64 x 28 x 28
    l = MaxPool2DDNNLayer(l, name='maxpool_7_3x3', pool_size=3, stride=(2, 2))
    # 64 x 13 x 13
    l = Conv2DDNNLayer(l, name='conv2d_8_3x3', num_filters=128, filter_size=(3, 3), **conv_kwargs)
    # 128 x 13 x 13
    l = Conv2DDNNLayer(l, name='conv2d_9_3x3', num_filters=128, filter_size=(3, 3), **conv_kwargs)
    # 128 x 13 x 13
    l = Conv2DDNNLayer(l, name='conv2d_10_3x3', num_filters=128, filter_size=(3, 3), **conv_kwargs)
    # 128 x 13 x 13
    l = MaxPool2DDNNLayer(l, name='maxpool_11_3x3', pool_size=3, stride=(2, 2))
    # 128 x 6 x 6
    l = Conv2DDNNLayer(l, name='conv2d_12_3x3', num_filters=256, filter_size=(3, 3), **conv_kwargs)
    # 256 x 6 x 6
    l = Conv2DDNNLayer(l, name='conv2d_13_3x3', num_filters=256, filter_size=(3, 3), **conv_kwargs)
    # 256 x 6 x 6
    l = Conv2DDNNLayer(l, name='conv2d_14_3x3', num_filters=256, filter_size=(3, 3), **conv_kwargs)
    # 256 x 6 x 6
    l = RMSPoolLayer(l, name='rmspool_18_3x3', pool_size=3, stride=(3, 3))
    # 256 x 2 x 2
    l = nn.layers.DropoutLayer(l, name='drop_19', p=0.5)
    # 256 x 2 x 2
    l = nn.layers.DenseLayer(l, name='dense_20', num_units=1024, **dense_kwargs)
    # 1024
    l = nn.layers.FeaturePoolLayer(l, name='featurepool_21', pool_size=2)
    # 512
    l = nn.layers.DropoutLayer(l, name='drop_22', p=0.5)
    # 512
    l = nn.layers.DenseLayer(l, name='dense_23', num_units=1024, **dense_kwargs)
    # 1024
    l = nn.layers.FeaturePoolLayer(l, name='featurepool_24', pool_size=2)
    # 512
    l = nn.layers.DenseLayer(l, name='out_25', num_units=output_num)

    return l


def build_448model(img_width, img_height, output_num, regression):
    conv_kwargs = dict(
        pad='same',
        nonlinearity=nn.nonlinearities.leaky_rectify,
        W=nn.init.Orthogonal(gain=1.0),
        b=nn.init.Constant(0.05),
        untie_biases=True,
    )
    dense_kwargs = dict(
        nonlinearity=nn.nonlinearities.leaky_rectify,
        W=nn.init.Orthogonal(1.0),
        b=nn.init.Constant(0.05),
    )

    l = nn.layers.InputLayer(name='in', shape=(None, 3, img_width, img_height))
    # 3 x 448 x 448
    l = Conv2DDNNLayer(l, name='conv2d_1_5x5', num_filters=32, filter_size=(5, 5), stride=(2, 2), **conv_kwargs)
    # 32 x 224 x 224
    l = Conv2DDNNLayer(l, name='conv2d_2_3x3', num_filters=32, filter_size=(3, 3), **conv_kwargs)
    # 32 x 224 x 224
    l = MaxPool2DDNNLayer(l, name='maxpool_3_3x3', pool_size=3, stride=(2, 2))
    # 32 x 111 x 111
    l = Conv2DDNNLayer(l, name='conv2d_4_5x5', num_filters=64, filter_size=(5, 5), stride=(2, 2), **conv_kwargs)
    # 64 x 56 x 56
    l = Conv2DDNNLayer(l, name='conv2d_5_3x3', num_filters=64, filter_size=(3, 3), **conv_kwargs)
    # 64 x 56 x 56
    l = Conv2DDNNLayer(l, name='conv2d_6_3x3', num_filters=64, filter_size=(3, 3), **conv_kwargs)
    # 64 x 56 x 56
    l = MaxPool2DDNNLayer(l, name='maxpool_7_3x3', pool_size=3, stride=(2, 2))
    # 64 x 27 x 27
    l = Conv2DDNNLayer(l, name='conv2d_8_3x3', num_filters=128, filter_size=(3, 3), **conv_kwargs)
    # 128 x 27 x 27
    l = Conv2DDNNLayer(l, name='conv2d_9_3x3', num_filters=128, filter_size=(3, 3), **conv_kwargs)
    # 128 x 27 x 27
    l = Conv2DDNNLayer(l, name='conv2d_10_3x3', num_filters=128, filter_size=(3, 3), **conv_kwargs)
    # 128 x 27 x 27
    l = MaxPool2DDNNLayer(l, name='maxpool_11_3x3', pool_size=3, stride=(2, 2))
    # 128 x 13 x 13
    l = Conv2DDNNLayer(l, name='conv2d_12_3x3', num_filters=256, filter_size=(3, 3), **conv_kwargs)
    # 256 x 13 x 13
    l = Conv2DDNNLayer(l, name='conv2d_13_3x3', num_filters=256, filter_size=(3, 3), **conv_kwargs)
    # 256 x 13 x 13
    l = Conv2DDNNLayer(l, name='conv2d_14_3x3', num_filters=256, filter_size=(3, 3), **conv_kwargs)
    # 256 x 13 x 13
    l = MaxPool2DDNNLayer(l, name='maxpool_15_3x3', pool_size=3, stride=(2, 2))
    # 256 x 6 x 6
    l = Conv2DDNNLayer(l, name='conv2d_16_3x3', num_filters=512, filter_size=(3, 3), **conv_kwargs)
    # 512 x 6 x 6
    l = Conv2DDNNLayer(l, name='conv2d_17_3x3', num_filters=512, filter_size=(3, 3), **conv_kwargs)
    # 512 x 6 x 6
    l = RMSPoolLayer(l, name='rmspool_18_3x3', pool_size=3, stride=(3, 3))
    # 512 x 2 x 2
    l = nn.layers.DropoutLayer(l, name='drop_19', p=0.5)
    # 512 x 2 x 2
    l = nn.layers.DenseLayer(l, name='dense_20', num_units=1024, **dense_kwargs)
    # 1024
    l = nn.layers.FeaturePoolLayer(l, name='featurepool_21', pool_size=2)
    # 512
    l = nn.layers.DropoutLayer(l, name='drop_22', p=0.5)
    # 512
    l = nn.layers.DenseLayer(l, name='dense_23', num_units=1024, **dense_kwargs)
    # 1024
    l = nn.layers.FeaturePoolLayer(l, name='featurepool_24', pool_size=2)
    # 512
    l = nn.layers.DenseLayer(l, name='out_25', num_units=output_num)

    return l