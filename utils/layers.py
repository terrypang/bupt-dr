import lasagne as nn
from lasagne.layers.dnn import Conv2DDNNLayer, Pool2DDNNLayer
from theano.sandbox.cuda import dnn
from theano import tensor as T


class RMSPoolLayer(Pool2DDNNLayer):
    """Use RMS as pooling function.
    from https://github.com/benanne/kaggle-ndsb/blob/master/tmp_dnn.py
    """
    def __init__(self, incoming, pool_size, stride=None, pad=(0, 0),
                 epsilon=1e-12, **kwargs):
        super(RMSPoolLayer, self).__init__(incoming, pool_size,  stride,
                                           pad, **kwargs)
        self.epsilon = epsilon
        del self.mode

    def get_output_for(self, input, *args, **kwargs):
        out = dnn.dnn_pool(T.sqr(input), self.pool_size, self.stride,
                           'average')
        return T.sqrt(out + self.epsilon)


def conv2dbn(l, name, **kwargs):
    l = nn.layers.dnn.Conv2DDNNLayer(
        l, name=name,
        **kwargs
    )
    l = nn.layers.dnn.batch_norm_dnn(l, name='%sbn' % name)
    return l


# Residual Network
def residual_block(layer, name, num_layers, num_filters,
                   bottleneck=False, bottleneck_factor=4,
                   filter_size=(3, 3), stride=1, pad='same',
                   W=nn.init.GlorotUniform(),
                   nonlinearity=nn.nonlinearities.rectify):
    conv = layer

    # When changing filter size or feature map size
    if (num_filters != layer.output_shape[1]) or (stride != 1):
        # Projection shortcut, aka option B
        layer = conv2dbn(
            layer, name='%s_shortcut' % name, num_filters=num_filters,
            filter_size=1, stride=stride, pad=0, nonlinearity=None, b=None
        )

    if bottleneck and num_layers < 3:
        raise ValueError('At least 3 layers is required for bottleneck configuration')

    for i in range(num_layers):
        if bottleneck:
            # Force then first and last layer to use 1x1 convolution
            if i == 0 or (i == (num_layers - 1)):
                actual_filter_size = (1, 1)
            else:
                actual_filter_size = filter_size

            # Only increase the filter size to the target size for
            # the last layer
            if i == (num_layers - 1):
                actual_num_filters = num_filters
            else:
                actual_num_filters = num_filters / bottleneck_factor
        else:
            actual_num_filters = num_filters
            actual_filter_size = filter_size

        conv = conv2dbn(
            conv, name='%s_%s' % (name, i), num_filters=actual_num_filters,
            filter_size=actual_filter_size, pad=pad, W=W,
            # Remove nonlinearity for the last conv layer
            nonlinearity=nonlinearity if (i < num_layers - 1) else None,
            # Only apply stride for the first conv layer
            stride=stride if i == 0 else 1
        )

    l = nn.layers.merge.ElemwiseSumLayer([conv, layer], name='%s_elemsum' % name)
    l = nn.layers.NonlinearityLayer(l, nonlinearity=nonlinearity, name='%s_elemsum_nl' % name)
    return l


# Inception Network
def inceptionA(input_layer, name, nfilt):
    l1 = conv2dbn(
        input_layer, name='%s_inceptA_1_1x1' % name,
        num_filters=nfilt[0][0], filter_size=1
    )

    l2 = conv2dbn(
        input_layer, name='%s_inceptA_2_1x1' % name,
        num_filters=nfilt[1][0], filter_size=1
    )
    l2 = conv2dbn(
        l2, name='%s_inceptA_2_5x5' % name,
        num_filters=nfilt[1][1], filter_size=5, pad=2
    )

    l3 = conv2dbn(
        input_layer, name='%s_inceptA_3_1x1' % name,
        num_filters=nfilt[2][0], filter_size=1
    )
    l3 = conv2dbn(
        l3, name='%s_inceptA_3_3x3_1' % name,
        num_filters=nfilt[2][1], filter_size=3, pad=1
    )
    l3 = conv2dbn(
        l3, name='%s_inceptA_3_3x3_2' % name,
        num_filters=nfilt[2][2], filter_size=3, pad=1
    )

    l4 = nn.layers.dnn.Pool2DDNNLayer(
        input_layer, name='%s_inceptE_4p' % name,
        pool_size=3, stride=1, pad=1, mode='average_exc_pad'
    )
    l4 = conv2dbn(
        l4, name='%s_inceptA_4_1x1' % name,
        num_filters=nfilt[3][0], filter_size=1
    )

    return nn.layers.ConcatLayer(
        [l1, l2, l3, l4], name='%s_inceptA_concat' % name
    )


def inceptionB(input_layer, name, nfilt):
    l1 = conv2dbn(
        input_layer, name='%s_inceptB_1_3x3' % name,
        num_filters=nfilt[0][0], filter_size=3, stride=2
    )

    l2 = conv2dbn(
        input_layer, name='%s_inceptB_2_1x1' % name,
        num_filters=nfilt[1][0], filter_size=1
    )
    l2 = conv2dbn(
        l2, name='%s_inceptB_2_3x3_1' % name,
        num_filters=nfilt[1][1], filter_size=3, pad=1
    )
    l2 = conv2dbn(
        l2, name='%s_inceptB_2_3x3_2' % name,
        num_filters=nfilt[1][2], filter_size=3, stride=2
    )

    l3 = nn.layers.dnn.Pool2DDNNLayer(
        input_layer, name='%s_inceptE_3p' % name,
        pool_size=3, stride=2, mode='average_exc_pad'
    )

    return nn.layers.ConcatLayer(
        [l1, l2, l3], name='%s_inceptB_concat' % name
    )


def inceptionC(input_layer, name, nfilt):
    l1 = conv2dbn(
        input_layer, name='%s_inceptC_1_3x3' % name,
        num_filters=nfilt[0][0], filter_size=1
    )

    l2 = conv2dbn(
        input_layer, name='%s_inceptC_2_3x3' % name,
        num_filters=nfilt[1][0], filter_size=1
    )
    l2 = conv2dbn(
        l2, name='%s_inceptC_2_1x7' % name,
        num_filters=nfilt[1][1], filter_size=(1, 7), pad=(0, 3)
    )
    l2 = conv2dbn(
        l2, name='%s_inceptC_2_7x1' % name,
        num_filters=nfilt[1][2], filter_size=(7, 1), pad=(3, 0)
    )

    l3 = conv2dbn(
        input_layer, name='%s_inceptC_3_1x1' % name,
        num_filters=nfilt[2][0], filter_size=1
    )
    l3 = conv2dbn(
        l3, name='%s_inceptC_3_7x1_1' % name,
        num_filters=nfilt[2][1], filter_size=(7, 1), pad=(3, 0)
    )
    l3 = conv2dbn(
        l3, name='%s_inceptC_3_1x7_1' % name,
        num_filters=nfilt[2][2], filter_size=(1, 7), pad=(0, 3)
    )
    l3 = conv2dbn(
        l3, name='%s_inceptC_3_7x1_2' % name,
        num_filters=nfilt[2][3], filter_size=(7, 1), pad=(3, 0)
    )
    l3 = conv2dbn(
        l3, name='%s_inceptC_3_1x7_2' % name,
        num_filters=nfilt[2][4], filter_size=(1, 7), pad=(0, 3)
    )

    l4 = nn.layers.dnn.Pool2DDNNLayer(
        input_layer, name='%s_inceptE_4p' % name,
        pool_size=3, stride=1, pad=1, mode='average_exc_pad'
    )
    l4 = conv2dbn(
        l4, name='%s_inceptC_4_1x1' % name,
        num_filters=nfilt[3][0], filter_size=1
    )

    return nn.layers.ConcatLayer(
        [l1, l2, l3, l4], name='%s_inceptC_concat' % name
    )


def inceptionD(input_layer, name, nfilt):
    l1 = conv2dbn(
        input_layer, name='%s_inceptD_1_1x1' % name,
        num_filters=nfilt[0][0], filter_size=1
    )
    l1 = conv2dbn(
        l1, name='%s_inceptD_1_3x3' % name,
        num_filters=nfilt[0][1], filter_size=3, stride=2
    )

    l2 = conv2dbn(
        input_layer, name='%s_inceptD_2_1x1' % name,
        num_filters=nfilt[1][0], filter_size=1
    )
    l2 = conv2dbn(
        l2, name='%s_inceptD_2_1x7' % name,
        num_filters=nfilt[1][1], filter_size=(1, 7), pad=(0, 3)
    )
    l2 = conv2dbn(
        l2, name='%s_inceptD_2_7x1' % name,
        num_filters=nfilt[1][2], filter_size=(7, 1), pad=(3, 0)
    )
    l2 = conv2dbn(
        l2, name='%s_inceptD_2_3x3' % name,
        num_filters=nfilt[1][3], filter_size=3, stride=2
    )

    l3 = nn.layers.dnn.Pool2DDNNLayer(
        input_layer, name='%s_inceptD_3p' % name,
        pool_size=3, stride=2, mode='max'
    )

    return nn.layers.ConcatLayer(
        [l1, l2, l3], name='%s_inceptD_concat' % name
    )


def inceptionE(input_layer, name, nfilt, pool_type):
    l1 = conv2dbn(
        input_layer, name='%s_inceptE_1_1x1' % name,
        num_filters=nfilt[0][0], filter_size=1
    )

    l2 = conv2dbn(
        input_layer, name='%s_inceptE_2_1x1' % name,
        num_filters=nfilt[1][0], filter_size=1
    )
    l2a = conv2dbn(
        l2, name='%s_inceptE_2a_1x3' % name,
        num_filters=nfilt[1][1], filter_size=(1, 3), pad=(0, 1)
    )
    l2b = conv2dbn(
        l2, name='%s_inceptE_2b_3x1' % name,
        num_filters=nfilt[1][2], filter_size=(3, 1), pad=(1, 0)
    )

    l3 = conv2dbn(
        input_layer, name='%s_inceptE_3_1x1_1' % name,
        num_filters=nfilt[2][0], filter_size=1
    )
    l3 = conv2dbn(
        l3, name='%s_inceptE_3_1x1_2' % name,
        num_filters=nfilt[2][1], filter_size=3, pad=1
    )
    l3a = conv2dbn(
        l3, name='%s_inceptE_3a_1x3' % name,
        num_filters=nfilt[2][2], filter_size=(1, 3), pad=(0, 1)
    )
    l3b = conv2dbn(
        l3, name='%s_inceptE_3b_3x1' % name,
        num_filters=nfilt[2][3], filter_size=(3, 1), pad=(1, 0)
    )

    if pool_type == 'avg':
        l4 = nn.layers.dnn.Pool2DDNNLayer(
            input_layer, name='%s_inceptE_4p' % name,
            pool_size=3, stride=1, pad=1, mode='average_exc_pad'
        )
    elif pool_type == 'max':
        l4 = nn.layers.dnn.Pool2DDNNLayer(
            input_layer, name='%s_inceptE_4p' % name,
            pool_size=3, stride=1, pad=1, mode='max'
        )
    else:
        raise ValueError('unrecognized pool_type')
    l4 = conv2dbn(
        l4, name='%s_inceptE_4_1x1' % name,
        num_filters=nfilt[3][0], filter_size=1
    )

    return nn.layers.ConcatLayer(
        [l1, l2a, l2b, l3a, l3b, l4], name='%s_inceptE_concat' % name
    )


# Dense Network
def dense_block(network, num_layers, growth_rate, dropout, name_prefix):
    # concatenated 3x3 convolutions
    for n in range(num_layers):
        conv = affine_relu_conv(network, channels=growth_rate,
                                filter_size=3, dropout=dropout,
                                name_prefix=name_prefix + '_l%02d' % (n + 1))
        conv = nn.layers.dnn.BatchNormDNNLayer(conv, name=name_prefix + '_l%02dbn' % (n + 1),
                              beta=None, gamma=None)
        network = nn.layers.ConcatLayer([network, conv], axis=1,
                              name=name_prefix + '_l%02d_join' % (n + 1))
    return network


def transition(network, dropout, name_prefix):
    # a transition 1x1 convolution followed by avg-pooling
    network = affine_relu_conv(network, channels=network.output_shape[1],
                               filter_size=1, dropout=dropout,
                               name_prefix=name_prefix)
    network = nn.layers.dnn.Pool2DDNNLayer(network, 2, mode='average_inc_pad',
                          name=name_prefix + '_pool')
    network = nn.layers.dnn.BatchNormDNNLayer(network, name=name_prefix + '_bn',
                             beta=None, gamma=None)
    return network


def affine_relu_conv(network, channels, filter_size, dropout, name_prefix):
    network = nn.layers.ScaleLayer(network, name=name_prefix + '_scale')
    network = nn.layers.BiasLayer(network, name=name_prefix + '_shift')
    network = nn.layers.NonlinearityLayer(network, nonlinearity=nn.nonlinearities.rectify,
                                name=name_prefix + '_relu')
    network = nn.layers.dnn.Conv2DDNNLayer(network, channels, filter_size, pad='same',
                          W=nn.init.HeNormal(gain='relu'),
                          b=None, nonlinearity=None,
                          name=name_prefix + '_conv')
    if dropout:
        network = nn.layers.DropoutLayer(network, dropout)
    return network


# class ApplyNonlinearity(nn.layers.Layer):
#
#     def __init__(self, input_layer, nonlinearity=nn.nonlinearities.softmax):
#         super(ApplyNonlinearity, self).__init__(input_layer)
#         self.nonlinearity = nonlinearity
#
#     def get_output_shape_for(self, input_shape):
#         return input_shape
#
#     def get_output_for(self, input, *args, **kwargs):
#         return self.nonlinearity(input)
