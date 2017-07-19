from lasagne.layers.dnn import Conv2DDNNLayer
import lasagne as nn
from utils.layers import dense_block, transition


def build_model(img_width, img_height, num_units, depth=21, first_output=16, growth_rate=12, num_blocks=2,
                   dropout=0):
    """
    Creates a DenseNet model in Lasagne.
    Parameters
    ----------
    input_shape : tuple
        The shape of the input layer, as ``(batchsize, channels, rows, cols)``.
        Any entry except ``channels`` can be ``None`` to indicate free size.
    input_var : Theano expression or None
        Symbolic input variable. Will be created automatically if not given.
    classes : int
        The number of classes of the softmax output.
    depth : int
        Depth of the network. Must be ``num_blocks * n + 1`` for some ``n``.
        (Parameterizing by depth rather than n makes it easier to follow the
        paper.)
    first_output : int
        Number of channels of initial convolution before entering the first
        dense block, should be of comparable size to `growth_rate`.
    growth_rate : int
        Number of feature maps added per layer.
    num_blocks : int
        Number of dense blocks (defaults to 3, as in the original paper).
    dropout : float
        The dropout rate. Set to zero (the default) to disable dropout.
    batchsize : int or None
        The batch size to build the model for, or ``None`` (the default) to
        allow any batch size.
    inputsize : int, tuple of int or None
    Returns
    -------
    network : Layer instance
        Lasagne Layer instance for the output layer.
    References
    ----------
    .. [1] Gao Huang et al. (2016):
           Densely Connected Convolutional Networks.
           https://arxiv.org/abs/1608.06993
    """
    if (depth - 1) % num_blocks != 0:
        raise ValueError("depth must be num_blocks * n + 1 for some n")

    # input and initial convolution
    network = nn.layers.InputLayer(shape=(None, 3, img_width, img_height), name='input')
    network = Conv2DDNNLayer(network, first_output, 3, pad='same',
                          W=nn.init.HeNormal(gain='relu'),
                          b=None, nonlinearity=None, name='pre_conv')
    network = nn.layers.dnn.BatchNormDNNLayer(network, name='pre_bn', beta=None, gamma=None)
    # note: The authors' implementation does *not* have a dropout after the
    #       initial convolution. This was missing in the paper, but important.
    # if dropout:
    #     network = DropoutLayer(network, dropout)
    # dense blocks with transitions in between
    n = (depth - 1) // num_blocks
    for b in range(num_blocks):
        network = dense_block(network, n - 1, growth_rate, dropout,
                              name_prefix='block%d' % (b + 1))
        if b < num_blocks - 1:
            network = transition(network, dropout,
                                 name_prefix='block%d_trs' % (b + 1))
    # post processing until prediction
    network = nn.layers.ScaleLayer(network, name='post_scale')
    network = nn.layers.BiasLayer(network, name='post_shift')
    network = nn.layers.NonlinearityLayer(network, nonlinearity=nn.nonlinearities.rectify,
                                name='post_relu')
    network = nn.layers.GlobalPoolLayer(network, name='post_pool')
    network = nn.layers.DenseLayer(network, num_units, nonlinearity=None,
                         W=nn.init.HeNormal(gain=1), name='output')
    network = nn.layers.NonlinearityLayer(network, nonlinearity=nn.nonlinearities.softmax)
    return network



