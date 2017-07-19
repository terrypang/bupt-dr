import lasagne as nn
from lasagne.layers.dnn import Conv2DDNNLayer, MaxPool2DDNNLayer


def build_model(depth, size, output_num, regression):
    dense_kwargs = dict(
        nonlinearity=nn.nonlinearities.leaky_rectify,
        W=nn.init.Orthogonal(1.0),
        b=nn.init.Constant(0.01),
    )
    conv_kwargs = dict(
        pad='same',
        nonlinearity=nn.nonlinearities.leaky_rectify,
        W=nn.init.GlorotNormal(),
        b=nn.init.Constant(0.1),
        untie_biases=True,
    )
    max_pool_kwargs = dict(
        pool_size=3,
        stride=2,
        pad=1,
    )

    l = nn.layers.InputLayer(name='in', shape=(None, depth, size, size))
    l = Conv2DDNNLayer(l, name='conv2d_1_5x5', num_filters=32, filter_size=(3, 3), **conv_kwargs)
    l = MaxPool2DDNNLayer(l, name='maxpool_2_3x3', **max_pool_kwargs)
    l = Conv2DDNNLayer(l, name='conv2d_3_3x3', num_filters=64, filter_size=(3, 3), **conv_kwargs)
    l = MaxPool2DDNNLayer(l, name='maxpool_4_3x3', **max_pool_kwargs)
    l = Conv2DDNNLayer(l, name='conv2d_5_3x3', num_filters=128, filter_size=(3, 3), **conv_kwargs)
    l = MaxPool2DDNNLayer(l, name='maxpool_6_3x3', **max_pool_kwargs)
    l = nn.layers.DenseLayer(l, name='dense_7', num_units=500, **dense_kwargs)
    l = nn.layers.DropoutLayer(l, name='drop_8', p=0.5)
    l = nn.layers.DenseLayer(l, name='dense_9', num_units=500, **dense_kwargs)
    l = nn.layers.FeaturePoolLayer(l, name='featurepool_10', pool_size=2)
    l = nn.layers.DenseLayer(l, name='out', num_units=output_num)
    return l