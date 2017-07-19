from lasagne.layers.dnn import Conv2DDNNLayer, MaxPool2DDNNLayer, Pool2DDNNLayer
from utils.layers import conv2dbn
import lasagne as nn


def build_model(img_width, img_height, output_num, regression):
    l = build_drnet_model(img_width, img_height, output_num, regression)
    return l


def build_pretrain1_model(img_width, img_height, output_num, regression):
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
    dense_kwargs = dict(
        nonlinearity=nn.nonlinearities.leaky_rectify,
        W=nn.init.Orthogonal(1.0),
        b=nn.init.Constant(0.05),
    )

    l = nn.layers.InputLayer(name='in', shape=(None, 3, img_width, img_height))
    # 3 x 128 x 128
    l = Conv2DDNNLayer(l, name='conv2d_1_5x5', num_filters=32, filter_size=(3, 3), stride=2, **conv_kwargs)
    # 32 x 64 x 64
    l = Conv2DDNNLayer(l, name='conv2d_2_3x3', num_filters=32, filter_size=(3, 3), **conv_kwargs)
    # 32 x 64 x 64
    l = MaxPool2DDNNLayer(l, name='maxpool_3_3x3', **max_pool_kwargs)
    # 32 x 32 x 32
    l = Conv2DDNNLayer(l, name='conv2d_4_3x3', num_filters=64, filter_size=(3, 3), stride=(2, 2), **conv_kwargs)
    # 64 x 16 x 16
    l = Conv2DDNNLayer(l, name='conv2d_5_3x3', num_filters=64, filter_size=(3, 3), **conv_kwargs)
    # 64 x 16 x 16
    l = Conv2DDNNLayer(l, name='conv2d_6_3x3', num_filters=64, filter_size=(3, 3), **conv_kwargs)
    # 64 x 16 x 16
    l = MaxPool2DDNNLayer(l, name='maxpool_7_3x3', **max_pool_kwargs)
    # 64 x 8 x 8
    l = Conv2DDNNLayer(l, name='conv2d_8_3x3', num_filters=128, filter_size=(3, 3), **conv_kwargs)
    # 128 x 8 x 8
    l = Conv2DDNNLayer(l, name='conv2d_9_3x3', num_filters=256, filter_size=(3, 3), **conv_kwargs)
    # 256 x 8 x 8
    l = Conv2DDNNLayer(l, name='conv2d_10_3x3', num_filters=512, filter_size=(3, 3), **conv_kwargs)
    # 512 x 8 x 8
    l = Pool2DDNNLayer(l, name='out_avgpool', pool_size=8, stride=1, mode='average_exc_pad')
    # 512 x 1 x 1
    l = nn.layers.DropoutLayer(l, name='drop_22', p=0.5)
    # 512 x 1 x 1
    l = nn.layers.DenseLayer(l, name='dense_23', num_units=1024, **dense_kwargs)
    # 1024
    l = nn.layers.FeaturePoolLayer(l, name='featurepool_24', pool_size=2)
    # 512
    l = nn.layers.DenseLayer(l, name='out_25', num_units=output_num)

    return l


def build_pretrain2_model(img_width, img_height, output_num, regression):
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
    dense_kwargs = dict(
        nonlinearity=nn.nonlinearities.leaky_rectify,
        W=nn.init.Orthogonal(1.0),
        b=nn.init.Constant(0.05),
    )

    l = nn.layers.InputLayer(name='in', shape=(None, 3, img_width, img_height))
    # 3 x 256 x 256
    l = Conv2DDNNLayer(l, name='conv2d_1_5x5', num_filters=32, filter_size=(3, 3), stride=(2, 2), **conv_kwargs)
    # 32 x 128 x 128
    l = Conv2DDNNLayer(l, name='conv2d_2_3x3', num_filters=32, filter_size=(3, 3), **conv_kwargs)
    # 32 x 128 x 128
    l = MaxPool2DDNNLayer(l, name='maxpool_3_3x3', **max_pool_kwargs)
    # 32 x 64 x 64
    l = Conv2DDNNLayer(l, name='conv2d_4_5x5', num_filters=64, filter_size=(3, 3), stride=(2, 2), **conv_kwargs)
    # 64 x 32 x 32
    l = Conv2DDNNLayer(l, name='conv2d_5_3x3', num_filters=64, filter_size=(3, 3), **conv_kwargs)
    # 64 x 32 x 32
    l = Conv2DDNNLayer(l, name='conv2d_6_3x3', num_filters=64, filter_size=(3, 3), **conv_kwargs)
    # 64 x 32 x 32
    l = MaxPool2DDNNLayer(l, name='maxpool_7_3x3', **max_pool_kwargs)
    # 64 x 16 x 16
    l = Conv2DDNNLayer(l, name='conv2d_8_3x3', num_filters=128, filter_size=(3, 3), **conv_kwargs)
    # 128 x 16 x 16
    l = Conv2DDNNLayer(l, name='conv2d_9_3x3', num_filters=128, filter_size=(3, 3), **conv_kwargs)
    # 128 x 16 x 16
    l = Conv2DDNNLayer(l, name='conv2d_10_3x3', num_filters=128, filter_size=(3, 3), **conv_kwargs)
    # 128 x 16 x 16
    l = MaxPool2DDNNLayer(l, name='maxpool_11_3x3', **max_pool_kwargs)
    # 128 x 8 x 8
    l = Conv2DDNNLayer(l, name='conv2d_12_3x3', num_filters=256, filter_size=(3, 3), **conv_kwargs)
    # 256 x 8 x 8
    l = Conv2DDNNLayer(l, name='conv2d_13_3x3', num_filters=512, filter_size=(3, 3), **conv_kwargs)
    # 512 x 8 x 8
    l = Conv2DDNNLayer(l, name='conv2d_14_3x3', num_filters=1024, filter_size=(3, 3), **conv_kwargs)
    # 1024 x 8 x 8
    l = Pool2DDNNLayer(l, name='out_avgpool', pool_size=8, stride=1, mode='average_exc_pad')
    # 1024 x 1 x 1
    l = nn.layers.DropoutLayer(l, name='drop_22', p=0.5)
    # 1024 x 1 x 1
    l = nn.layers.DenseLayer(l, name='dense_23', num_units=1024, **dense_kwargs)
    # 1024
    l = nn.layers.FeaturePoolLayer(l, name='featurepool_24', pool_size=2)
    # 512
    l = nn.layers.DenseLayer(l, name='out_25', num_units=output_num)

    return l


def build_pretrain3_model(img_width, img_height, output_num, regression):
    conv_kwargs = dict(
        pad='same',
        nonlinearity=nn.nonlinearities.leaky_rectify,
        W=nn.init.HeNormal(),
        b=nn.init.Constant(0.1),
        untie_biases=True,
    )
    max_pool_kwargs = dict(
        pool_size=3,
        stride=2,
        pad=1,
    )
    dense_kwargs = dict(
        nonlinearity=nn.nonlinearities.leaky_rectify,
        W=nn.init.Orthogonal(1.0),
        b=nn.init.Constant(0.05),
    )

    l = nn.layers.InputLayer(name='in', shape=(None, 3, img_width, img_height))
    # 3 x 512 x 512
    l = Conv2DDNNLayer(l, name='conv2d_3x3_1', num_filters=32, filter_size=(3, 3), stride=(2, 2), **conv_kwargs)
    # 32 x 256 x 256
    l = Conv2DDNNLayer(l, name='conv2d_3x3_2', num_filters=32, filter_size=(3, 3), **conv_kwargs)
    # 32 x 256 x 256
    l = Conv2DDNNLayer(l, name='conv2d_3x3_3', num_filters=64, filter_size=(3, 3), **conv_kwargs)
    # 64 x 256 x 256
    l = Conv2DDNNLayer(l, name='conv2d_3x3_4', num_filters=64, filter_size=(3, 3), stride=(2, 2), **conv_kwargs)
    # 64 x 128 x 128
    l = Conv2DDNNLayer(l, name='conv2d_3x3_5', num_filters=64, filter_size=(3, 3), **conv_kwargs)
    # 64 x 128 x 128
    l = Conv2DDNNLayer(l, name='conv2d_3x3_6', num_filters=128, filter_size=(3, 3), **conv_kwargs)
    # 128 x 128 x 128
    l = MaxPool2DDNNLayer(l, name='maxpool_3x3_7', **max_pool_kwargs)
    # 128 x 64 x 64
    l = Conv2DDNNLayer(l, name='conv2d_3x3_8', num_filters=192, filter_size=(3, 3), **conv_kwargs)
    # 192 x 64 x 64
    l = Conv2DDNNLayer(l, name='conv2d_3x3_9', num_filters=192, filter_size=(3, 3), **conv_kwargs)
    # 192 x 64 x 64
    l = MaxPool2DDNNLayer(l, name='maxpool_3x3_10', **max_pool_kwargs)
    # 192 x 32 X 32

    l = Conv2DDNNLayer(l, name='conv2d_6_3x3', num_filters=192, filter_size=(3, 3), **conv_kwargs)
    # 192 x 32 x 32
    l = MaxPool2DDNNLayer(l, name='maxpool_7_3x3', **max_pool_kwargs)
    # 192 x 16 x 16
    l = Conv2DDNNLayer(l, name='conv2d_8_3x3', num_filters=256, filter_size=(3, 3), **conv_kwargs)
    # 256 x 16 x 16
    l = MaxPool2DDNNLayer(l, name='maxpool_11_3x3', **max_pool_kwargs)
    # 256 x 8 x 8
    l = Conv2DDNNLayer(l, name='conv2d_12_3x3', num_filters=512, filter_size=(3, 3), **conv_kwargs)
    # 512 x 8 x 8
    l = Conv2DDNNLayer(l, name='conv2d_13_3x3', num_filters=1024, filter_size=(3, 3), **conv_kwargs)
    # 1024 x 8 x 8
    l = Conv2DDNNLayer(l, name='conv2d_14_3x3', num_filters=2048, filter_size=(3, 3), **conv_kwargs)
    # 2048 x 8 x 8

    out = Pool2DDNNLayer(l, name='out_avgpool', pool_size=8, stride=1, mode='average_exc_pad')
    # 2048 x 1 X 1
    if regression:
        # out = nn.layers.DropoutLayer(out, name='out_drop', p=0.5)
        # 2048 x 1 x 1
        out = Conv2DDNNLayer(out, name='out_conv', num_filters=1024, filter_size=(1, 1), nonlinearity=nn.nonlinearities.leaky_rectify)
        # 1024 x 1 x 1
        out = nn.layers.FeaturePoolLayer(out, name='out_featurepool', pool_size=2)
        # 512 x 1 x 1
        out = nn.layers.DenseLayer(out, name='out', num_units=output_num)
    else:
        out = Conv2DDNNLayer(out, name='out_conv', num_filters=output_num, filter_size=(1, 1), nonlinearity=None)
        out = nn.layers.FlattenLayer(out, name='out')
        out = nn.layers.NonlinearityLayer(out, name='softmax', nonlinearity=nn.nonlinearities.softmax)

    return out


def build_drnet_model(img_width, img_height, output_num, regression):
    conv_kwargs = dict(
        pad='same',
        nonlinearity=nn.nonlinearities.leaky_rectify,
        W=nn.init.HeNormal(),
        b=nn.init.Constant(0.1),
        untie_biases=True,
    )
    max_pool_kwargs = dict(
        pool_size=3,
        stride=2,
        pad=1,
    )
    avg_pool_kwargs = dict(
        pool_size=3,
        stride=1,
        pad=1,
        mode='average_exc_pad',
    )
    dense_kwargs = dict(
        nonlinearity=nn.nonlinearities.leaky_rectify,
        W=nn.init.Orthogonal(1.0),
        b=nn.init.Constant(0.05),
    )

    l = nn.layers.InputLayer(name='in', shape=(None, 3, img_width, img_height))
    # 3 x 512 x 512
    l = Conv2DDNNLayer(l, name='conv2d_3x3_1', num_filters=32, filter_size=(3, 3), stride=(2, 2), **conv_kwargs)
    # 32 x 256 x 256
    l = Conv2DDNNLayer(l, name='conv2d_3x3_2', num_filters=32, filter_size=(3, 3), **conv_kwargs)
    # 32 x 256 x 256
    l = Conv2DDNNLayer(l, name='conv2d_3x3_3', num_filters=64, filter_size=(3, 3), **conv_kwargs)
    # 64 x 256 x 256
    l = Conv2DDNNLayer(l, name='conv2d_3x3_4', num_filters=64, filter_size=(3, 3), stride=(2, 2), **conv_kwargs)
    # 64 x 128 x 128
    l = Conv2DDNNLayer(l, name='conv2d_3x3_5', num_filters=64, filter_size=(3, 3), **conv_kwargs)
    # 64 x 128 x 128
    l = Conv2DDNNLayer(l, name='conv2d_3x3_6', num_filters=128, filter_size=(3, 3), **conv_kwargs)
    # 128 x 128 x 128
    l = MaxPool2DDNNLayer(l, name='maxpool_3x3_7', **max_pool_kwargs)
    # 128 x 64 x 64
    l = Conv2DDNNLayer(l, name='conv2d_3x3_8', num_filters=192, filter_size=(3, 3), **conv_kwargs)
    # 192 x 64 x 64
    l = Conv2DDNNLayer(l, name='conv2d_3x3_9', num_filters=192, filter_size=(3, 3), **conv_kwargs)
    # 192 x 64 x 64
    l = MaxPool2DDNNLayer(l, name='maxpool_3x3_10', **max_pool_kwargs)
    # 192 x 32 X 32

    # branch block 1
    # 192 x 32 X 32
    a_branch0 = Conv2DDNNLayer(l, name='mixed_a_branch_0_1x1', num_filters=64, filter_size=(1, 1), **conv_kwargs)
    a_branch1 = Conv2DDNNLayer(l, name='mixed_a_branch_1_1x1', num_filters=48, filter_size=(1, 1), **conv_kwargs)
    a_branch1 = Conv2DDNNLayer(a_branch1, name='mixed_a_branch_1_5x5', num_filters=64, filter_size=(5, 5), **conv_kwargs)
    a_branch2 = Conv2DDNNLayer(l, name='mixed_a_branch_2_1x1', num_filters=64, filter_size=(1, 1), **conv_kwargs)
    a_branch2 = Conv2DDNNLayer(a_branch2, name='mixed_a_branch_2_3x3a', num_filters=96, filter_size=(3, 3), **conv_kwargs)
    a_branch2 = Conv2DDNNLayer(a_branch2, name='mixed_a_branch_2_3x3b', num_filters=96, filter_size=(3, 3), **conv_kwargs)
    a_branch3 = Pool2DDNNLayer(l, name='mixed_a_branch_3_avgpool_3x3', **avg_pool_kwargs)
    a_branch3 = Conv2DDNNLayer(a_branch3, name='mixed_a_branch_3_1x1', num_filters=32, filter_size=(1, 1), **conv_kwargs)
    a_branch_out = nn.layers.ConcatLayer([a_branch0, a_branch1, a_branch2, a_branch3], name='mixed_a_out')
    # 256 x 32 X 32

    # branch block 1
    # 256 x 32 X 32
    b_branch0 = Conv2DDNNLayer(a_branch_out, name='mixed_b_branch_0_1x1', num_filters=64, filter_size=(1, 1), **conv_kwargs)
    b_branch1 = Conv2DDNNLayer(a_branch_out, name='mixed_b_branch_1_1x1', num_filters=48, filter_size=(1, 1), **conv_kwargs)
    b_branch1 = Conv2DDNNLayer(b_branch1, name='mixed_b_branch_1_5x5', num_filters=64, filter_size=(5, 5), **conv_kwargs)
    b_branch2 = Conv2DDNNLayer(a_branch_out, name='mixed_b_branch_2_1x1', num_filters=64, filter_size=(1, 1), **conv_kwargs)
    b_branch2 = Conv2DDNNLayer(b_branch2, name='mixed_b_branch_2_3x3a', num_filters=96, filter_size=(3, 3), **conv_kwargs)
    b_branch2 = Conv2DDNNLayer(b_branch2, name='mixed_b_branch_2_3x3b', num_filters=96, filter_size=(3, 3), **conv_kwargs)
    b_branch3 = Pool2DDNNLayer(a_branch_out, name='mixed_b_branch_3_avgpool_3x3', **avg_pool_kwargs)
    b_branch3 = Conv2DDNNLayer(b_branch3, name='mixed_b_branch_3_1x1', num_filters=64, filter_size=(1, 1), **conv_kwargs)
    b_branch_out = nn.layers.ConcatLayer([b_branch0, b_branch1, b_branch2, b_branch3], name='mixed_b_out')
    # 288 x 32 X 32

    # branch block 1
    # 288 x 32 X 32
    bc_branch0 = Conv2DDNNLayer(a_branch_out, name='mixed_bc_branch_0_1x1', num_filters=64, filter_size=(1, 1), **conv_kwargs)
    bc_branch1 = Conv2DDNNLayer(a_branch_out, name='mixed_bc_branch_1_1x1', num_filters=48, filter_size=(1, 1), **conv_kwargs)
    bc_branch1 = Conv2DDNNLayer(bc_branch1, name='mixed_bc_branch_1_5x5', num_filters=64, filter_size=(5, 5), **conv_kwargs)
    bc_branch2 = Conv2DDNNLayer(b_branch_out, name='mixed_bc_branch_2_1x1', num_filters=64, filter_size=(1, 1), **conv_kwargs)
    bc_branch2 = Conv2DDNNLayer(bc_branch2, name='mixed_bc_branch_2_3x3a', num_filters=96, filter_size=(3, 3), **conv_kwargs)
    bc_branch2 = Conv2DDNNLayer(bc_branch2, name='mixed_bc_branch_2_3x3b', num_filters=96, filter_size=(3, 3), **conv_kwargs)
    bc_branch3 = Pool2DDNNLayer(b_branch_out, name='mixed_bc_branch_3_avgpool_3x3', **avg_pool_kwargs)
    bc_branch3 = Conv2DDNNLayer(bc_branch3, name='mixed_bc_branch_3_1x1', num_filters=64, filter_size=(1, 1), **conv_kwargs)
    bc_branch_out = nn.layers.ConcatLayer([bc_branch0, bc_branch1, bc_branch2, bc_branch3], name='mixed_bc_out')
    # 288 x 32 X 32
    b_out = nn.layers.merge.ElemwiseSumLayer([b_branch_out, bc_branch_out], name='b_bc_elemsum')
    b_out = nn.layers.NonlinearityLayer(b_out, nonlinearity=nn.nonlinearities.leaky_rectify, name='b_bc_elemsum_nl')
    b_out = nn.layers.DropoutLayer(b_out, name='b_out_drop', p=0.1)

    # branch block 2
    # 288 x 32 X 32
    c_branch0 = Conv2DDNNLayer(b_out, name='mixed_c_branch_0_3x3', num_filters=384, filter_size=(3, 3), stride=2, **conv_kwargs)
    c_branch1 =  Conv2DDNNLayer(b_out, name='mixed_c_branch_1_1x1', num_filters=64, filter_size=(1, 1), **conv_kwargs)
    c_branch1 = Conv2DDNNLayer(c_branch1, name='mixed_c_branch_1_3x3a', num_filters=96, filter_size=(3, 3), **conv_kwargs)
    c_branch1 = Conv2DDNNLayer(c_branch1, name='mixed_c_branch_1_3x3b', num_filters=96, filter_size=(3, 3), stride=2, **conv_kwargs)
    c_branch2 = MaxPool2DDNNLayer(b_out, name='mixed_c_branch_2_maxpool_3x3', **max_pool_kwargs)
    c_branch_out = nn.layers.ConcatLayer([c_branch0, c_branch1, c_branch2], name='mixed_c_out')
    # 768 x 16 X 16

    # branch block 2
    # 768 x 16 X 16
    cd1_branch0 = Conv2DDNNLayer(c_branch_out, name='mixed_cd1_branch_0_1x1', num_filters=192, filter_size=(1, 1), **conv_kwargs)
    cd1_branch1 = Conv2DDNNLayer(c_branch_out, name='mixed_cd1_branch_1_1x1', num_filters=128, filter_size=(1, 1), **conv_kwargs)
    cd1_branch1 = Conv2DDNNLayer(cd1_branch1, name='mixed_cd1_branch_1_1x7', num_filters=128, filter_size=(1, 7), **conv_kwargs)
    cd1_branch1 = Conv2DDNNLayer(cd1_branch1, name='mixed_cd1_branch_1_7x1', num_filters=192, filter_size=(7, 1), **conv_kwargs)
    cd1_branch2 = Conv2DDNNLayer(c_branch_out, name='mixed_cd1_branch_2_1x1', num_filters=128, filter_size=(1, 1), **conv_kwargs)
    cd1_branch2 = Conv2DDNNLayer(cd1_branch2, name='mixed_cd1_branch_2_7x1a', num_filters=128, filter_size=(7, 1), **conv_kwargs)
    cd1_branch2 = Conv2DDNNLayer(cd1_branch2, name='mixed_cd1_branch_2_1x7a', num_filters=128, filter_size=(1, 7), **conv_kwargs)
    cd1_branch2 = Conv2DDNNLayer(cd1_branch2, name='mixed_cd1_branch_2_7x1b', num_filters=128, filter_size=(7, 1), **conv_kwargs)
    cd1_branch2 = Conv2DDNNLayer(cd1_branch2, name='mixed_cd1_branch_2_1x7b', num_filters=192, filter_size=(1, 7), **conv_kwargs)
    cd1_branch3 = Pool2DDNNLayer(c_branch_out, name='mixed_cd1_branch_3_avgpool_3x3', **avg_pool_kwargs)
    cd1_branch3 = Conv2DDNNLayer(cd1_branch3, name='mixed_cd1_branch_3_1x1', num_filters=192, filter_size=(1, 1), **conv_kwargs)
    cd1_branch_out = nn.layers.ConcatLayer([cd1_branch0, cd1_branch1, cd1_branch2, cd1_branch3], name='mixed_cd1_out')
    # 768 x 16 X 16

    # branch block 2
    # 768 x 16 X 16
    cd2_branch0 = Conv2DDNNLayer(c_branch_out, name='mixed_cd2_branch_0_1x1', num_filters=192, filter_size=(1, 1), **conv_kwargs)
    cd2_branch1 = Conv2DDNNLayer(c_branch_out, name='mixed_cd2_branch_1_1x1', num_filters=160, filter_size=(1, 1), **conv_kwargs)
    cd2_branch1 = Conv2DDNNLayer(cd2_branch1, name='mixed_cd2_branch_1_1x7', num_filters=160, filter_size=(1, 7), **conv_kwargs)
    cd2_branch1 = Conv2DDNNLayer(cd2_branch1, name='mixed_cd2_branch_1_7x1', num_filters=192, filter_size=(7, 1), **conv_kwargs)
    cd2_branch2 = Conv2DDNNLayer(cd1_branch_out, name='mixed_cd2_branch_2_1x1', num_filters=160, filter_size=(1, 1), **conv_kwargs)
    cd2_branch2 = Conv2DDNNLayer(cd2_branch2, name='mixed_cd2_branch_2_7x1a', num_filters=160, filter_size=(7, 1), **conv_kwargs)
    cd2_branch2 = Conv2DDNNLayer(cd2_branch2, name='mixed_cd2_branch_2_1x7a', num_filters=160, filter_size=(1, 7), **conv_kwargs)
    cd2_branch2 = Conv2DDNNLayer(cd2_branch2, name='mixed_cd2_branch_2_7x1b', num_filters=160, filter_size=(7, 1), **conv_kwargs)
    cd2_branch2 = Conv2DDNNLayer(cd2_branch2, name='mixed_cd2_branch_2_1x7b', num_filters=192, filter_size=(1, 7), **conv_kwargs)
    cd2_branch3 = Pool2DDNNLayer(cd1_branch_out, name='mixed_cd2_branch_3_avgpool_3x3', **avg_pool_kwargs)
    cd2_branch3 = Conv2DDNNLayer(cd2_branch3, name='mixed_cd2_branch_3_1x1', num_filters=192, filter_size=(1, 1), **conv_kwargs)
    cd2_branch_out = nn.layers.ConcatLayer([cd2_branch0, cd2_branch1, cd2_branch2, cd2_branch3], name='mixed_cd2_out')
    # 768 x 16 X 16

    # branch block 2
    # 768 x 16 X 16
    cd3_branch0 = Conv2DDNNLayer(cd1_branch_out, name='mixed_cd3_branch_0_1x1', num_filters=192, filter_size=(1, 1), **conv_kwargs)
    cd3_branch1 = Conv2DDNNLayer(cd1_branch_out, name='mixed_cd3_branch_1_1x1', num_filters=160, filter_size=(1, 1), **conv_kwargs)
    cd3_branch1 = Conv2DDNNLayer(cd3_branch1, name='mixed_cd3_branch_1_1x7', num_filters=160, filter_size=(1, 7), **conv_kwargs)
    cd3_branch1 = Conv2DDNNLayer(cd3_branch1, name='mixed_cd3_branch_1_7x1', num_filters=192, filter_size=(7, 1), **conv_kwargs)
    cd3_branch2 = Conv2DDNNLayer(cd2_branch_out, name='mixed_cd3_branch_2_1x1', num_filters=160, filter_size=(1, 1), **conv_kwargs)
    cd3_branch2 = Conv2DDNNLayer(cd3_branch2, name='mixed_cd3_branch_2_7x1a', num_filters=160, filter_size=(7, 1), **conv_kwargs)
    cd3_branch2 = Conv2DDNNLayer(cd3_branch2, name='mixed_cd3_branch_2_1x7a', num_filters=160, filter_size=(1, 7), **conv_kwargs)
    cd3_branch2 = Conv2DDNNLayer(cd3_branch2, name='mixed_cd3_branch_2_7x1b', num_filters=160, filter_size=(7, 1), **conv_kwargs)
    cd3_branch2 = Conv2DDNNLayer(cd3_branch2, name='mixed_cd3_branch_2_1x7b', num_filters=192, filter_size=(1, 7), **conv_kwargs)
    cd3_branch3 = Pool2DDNNLayer(cd2_branch_out, name='mixed_cd3_branch_3_avgpool_3x3', **avg_pool_kwargs)
    cd3_branch3 = Conv2DDNNLayer(cd3_branch3, name='mixed_cd3_branch_3_1x1', num_filters=192, filter_size=(1, 1), **conv_kwargs)
    cd3_branch_out = nn.layers.ConcatLayer([cd3_branch0, cd3_branch1, cd3_branch2, cd3_branch3], name='mixed_cd3_out')
    # 768 x 16 X 16

    # branch block 2
    # 768 x 16 X 16
    cd4_branch0 = Conv2DDNNLayer(c_branch_out, name='mixed_cd4_branch_0_1x1', num_filters=192, filter_size=(1, 1), **conv_kwargs)
    cd4_branch1 = Conv2DDNNLayer(cd1_branch_out, name='mixed_cd4_branch_1_1x1', num_filters=192, filter_size=(1, 1), **conv_kwargs)
    cd4_branch1 = Conv2DDNNLayer(cd4_branch1, name='mixed_cd4_branch_1_1x7', num_filters=192, filter_size=(1, 7), **conv_kwargs)
    cd4_branch1 = Conv2DDNNLayer(cd4_branch1, name='mixed_cd4_branch_1_7x1', num_filters=192, filter_size=(7, 1), **conv_kwargs)
    cd4_branch2 = Conv2DDNNLayer(cd2_branch_out, name='mixed_cd4_branch_2_1x1', num_filters=192, filter_size=(1, 1), **conv_kwargs)
    cd4_branch2 = Conv2DDNNLayer(cd4_branch2, name='mixed_cd4_branch_2_7x1a', num_filters=192, filter_size=(7, 1), **conv_kwargs)
    cd4_branch2 = Conv2DDNNLayer(cd4_branch2, name='mixed_cd4_branch_2_1x7a', num_filters=192, filter_size=(1, 7), **conv_kwargs)
    cd4_branch2 = Conv2DDNNLayer(cd4_branch2, name='mixed_cd4_branch_2_7x1b', num_filters=192, filter_size=(7, 1), **conv_kwargs)
    cd4_branch2 = Conv2DDNNLayer(cd4_branch2, name='mixed_cd4_branch_2_1x7b', num_filters=192, filter_size=(1, 7), **conv_kwargs)
    cd4_branch3 = Pool2DDNNLayer(cd3_branch_out, name='mixed_cd4_branch_3_avgpool_3x3', **avg_pool_kwargs)
    cd4_branch3 = Conv2DDNNLayer(cd4_branch3, name='mixed_cd4_branch_3_1x1', num_filters=192, filter_size=(1, 1), **conv_kwargs)
    cd4_branch_out = nn.layers.ConcatLayer([cd4_branch0, cd4_branch1, cd4_branch2, cd4_branch3], name='mixed_cd4_out')
    # 768 x 16 X 16
    c_out = nn.layers.merge.ElemwiseSumLayer([c_branch_out, cd4_branch_out], name='c_cd4_elemsum')
    c_out = nn.layers.NonlinearityLayer(c_out, nonlinearity=nn.nonlinearities.leaky_rectify, name='c_cd4_elemsum_nl')
    c_out = nn.layers.DropoutLayer(c_out, name='c_out_drop', p=0.2)

    # branch block 3
    # 768 x 16 X 16
    d_branch0 = Conv2DDNNLayer(c_out, name='mixed_d_branch_0_1x1', num_filters=192, filter_size=(1, 1), **conv_kwargs)
    d_branch0 = Conv2DDNNLayer(d_branch0, name='mixed_d_branch_0_3x3', num_filters=320, filter_size=(3, 3), stride=2, **conv_kwargs)
    d_branch1 = Conv2DDNNLayer(c_out, name='mixed_d_branch_1_1x1', num_filters=192, filter_size=(1, 1), **conv_kwargs)
    d_branch1 = Conv2DDNNLayer(d_branch1, name='mixed_d_branch_1_1x7', num_filters=192, filter_size=(1, 7), **conv_kwargs)
    d_branch1 = Conv2DDNNLayer(d_branch1, name='mixed_d_branch_1_7x1', num_filters=192, filter_size=(7, 1), **conv_kwargs)
    d_branch1 = Conv2DDNNLayer(d_branch1, name='mixed_d_branch_1_3x3', num_filters=192, filter_size=(3, 3), stride=2, **conv_kwargs)
    d_branch2 = MaxPool2DDNNLayer(c_out, name='mixed_d_branch_2_maxpool_1x1', **max_pool_kwargs)
    d_branch_out = nn.layers.ConcatLayer([d_branch0, d_branch1, d_branch2], name='mixed_d_out')
    # 1280 x 8 X 8

    # branch block 3
    # 1280 x 8 X 8
    e_branch0 = Conv2DDNNLayer(d_branch_out, name='mixed_e_branch_0_1x1', num_filters=320, filter_size=(1, 1), **conv_kwargs)
    e_branch1 = Conv2DDNNLayer(d_branch_out, name='mixed_e_branch_1_1x1', num_filters=384, filter_size=(1, 1), **conv_kwargs)
    e_branch1_1 = Conv2DDNNLayer(e_branch1, name='mixed_e_branch_1_1x3', num_filters=384, filter_size=(1, 3), **conv_kwargs)
    e_branch1_2 = Conv2DDNNLayer(e_branch1, name='mixed_e_branch_1_3x1', num_filters=384, filter_size=(3, 1), **conv_kwargs)
    e_branch1_concat = nn.layers.ConcatLayer([e_branch1_1, e_branch1_2], name='mixed_e_branch_1_concat')
    e_branch2 = Conv2DDNNLayer(d_branch_out, name='mixed_e_branch_2_1x1', num_filters=448, filter_size=(1, 1), **conv_kwargs)
    e_branch2 = Conv2DDNNLayer(e_branch2, name='mixed_e_branch_2_3x3', num_filters=384, filter_size=(3, 3), **conv_kwargs)
    e_branch2_1 = Conv2DDNNLayer(e_branch2, name='mixed_e_branch_2_1x3', num_filters=384, filter_size=(1, 3), **conv_kwargs)
    e_branch2_2 = Conv2DDNNLayer(e_branch2, name='mixed_e_branch_2_3x1', num_filters=384, filter_size=(3, 1), **conv_kwargs)
    e_branch2_concat = nn.layers.ConcatLayer([e_branch2_1, e_branch2_2], name='mixed_e_branch_2_concat')
    e_branch3 = Pool2DDNNLayer(d_branch_out, name='mixed_e_branch_3_avgpool_3x3', **avg_pool_kwargs)
    e_branch3 = Conv2DDNNLayer(e_branch3, name='mixed_e_branch_3_1x1', num_filters=192, filter_size=(1, 1), **conv_kwargs)
    e_branch_out = nn.layers.ConcatLayer([e_branch0, e_branch1_concat, e_branch2_concat, e_branch3], name='mixed_e_out')
    # 2048 x 8 X 8

    # branch block 3
    # 2048 x 8 X 8
    eo_branch0 = Conv2DDNNLayer(d_branch_out, name='mixed_eo_branch_0_1x1', num_filters=320, filter_size=(1, 1), **conv_kwargs)
    eo_branch1 = Conv2DDNNLayer(d_branch_out, name='mixed_eo_branch_1_1x1', num_filters=384, filter_size=(1, 1), **conv_kwargs)
    eo_branch1_1 = Conv2DDNNLayer(eo_branch1, name='mixed_eo_branch_1_1x3', num_filters=384, filter_size=(1, 3), **conv_kwargs)
    eo_branch1_2 = Conv2DDNNLayer(eo_branch1, name='mixed_eo_branch_1_3x1', num_filters=384, filter_size=(3, 1), **conv_kwargs)
    eo_branch1_concat = nn.layers.ConcatLayer([eo_branch1_1, eo_branch1_2], name='mixed_eo_branch_1_concat')
    eo_branch2 = Conv2DDNNLayer(e_branch_out, name='mixed_eo_branch_2_1x1', num_filters=448, filter_size=(1, 1), **conv_kwargs)
    eo_branch2 = Conv2DDNNLayer(eo_branch2, name='mixed_eo_branch_2_3x3', num_filters=384, filter_size=(3, 3), **conv_kwargs)
    eo_branch2_1 = Conv2DDNNLayer(eo_branch2, name='mixed_eo_branch_2_1x3', num_filters=384, filter_size=(1, 3), **conv_kwargs)
    eo_branch2_2 = Conv2DDNNLayer(eo_branch2, name='mixed_eo_branch_2_3x1', num_filters=384, filter_size=(3, 1), **conv_kwargs)
    eo_branch2_concat = nn.layers.ConcatLayer([eo_branch2_1, eo_branch2_2], name='mixed_eo_branch_2_concat')
    eo_branch3 = Pool2DDNNLayer(e_branch_out, name='mixed_eo_branch_3_avgpool_3x3', **avg_pool_kwargs)
    eo_branch3 = Conv2DDNNLayer(eo_branch3, name='mixed_eo_branch_3_1x1', num_filters=192, filter_size=(1, 1), **conv_kwargs)
    eo_branch_out = nn.layers.ConcatLayer([eo_branch0, eo_branch1_concat, eo_branch2_concat, eo_branch3], name='mixed_eo_out')
    # 2048 x 8 X 8
    e_out = nn.layers.merge.ElemwiseSumLayer([e_branch_out, eo_branch_out], name='e_eo_elemsum')
    e_out = nn.layers.NonlinearityLayer(e_out, nonlinearity=nn.nonlinearities.leaky_rectify, name='e_eo_elemsum_nl')
    e_out = nn.layers.DropoutLayer(e_out, name='e_out_drop', p=0.3)

    # output
    # 2048 x 8 X 8
    out = Pool2DDNNLayer(e_out, name='out_avgpool', pool_size=8, stride=1, mode='average_exc_pad')
    # 2048 x 1 X 1
    if regression:
        out = nn.layers.DropoutLayer(out, name='out_drop', p=0.5)
        # 2048 x 1 x 1
        out = nn.layers.DenseLayer(out, name='out_dense', num_units=1024, **dense_kwargs)
        # 1024
        out = nn.layers.FeaturePoolLayer(out, name='out_featurepool', pool_size=2)
        # 512
        out = nn.layers.DenseLayer(out, name='out', num_units=output_num)
    else:
        out = Conv2DDNNLayer(out, name='out_conv', num_filters=output_num, filter_size=(1, 1), nonlinearity=None)
        out = nn.layers.FlattenLayer(out, name='out')
        out = nn.layers.NonlinearityLayer(out, name='softmax', nonlinearity=nn.nonlinearities.softmax)

    return out


def build_drnet_bn_model(img_width, img_height, output_num, regression):
    conv_kwargs = dict(
        pad='same',
        nonlinearity=nn.nonlinearities.leaky_rectify,
        W=nn.init.HeNormal(),
        b=nn.init.Constant(0.1),
        untie_biases=True,
    )
    max_pool_kwargs = dict(
        pool_size=3,
        stride=2,
        pad=1,
    )
    avg_pool_kwargs = dict(
        pool_size=3,
        stride=1,
        pad=1,
        mode='average_exc_pad',
    )
    dense_kwargs = dict(
        nonlinearity=nn.nonlinearities.leaky_rectify,
        W=nn.init.Orthogonal(1.0),
        b=nn.init.Constant(0.05),
    )

    l = nn.layers.InputLayer(name='in', shape=(None, 3, img_width, img_height))
    # 3 x 512 x 512
    l = Conv2DDNNLayer(l, name='conv2d_3x3_1', num_filters=32, filter_size=(3, 3), stride=(2, 2), **conv_kwargs)
    # 32 x 256 x 256
    l = Conv2DDNNLayer(l, name='conv2d_3x3_2', num_filters=32, filter_size=(3, 3), **conv_kwargs)
    # 32 x 256 x 256
    l = Conv2DDNNLayer(l, name='conv2d_3x3_3', num_filters=64, filter_size=(3, 3), **conv_kwargs)
    # 64 x 256 x 256
    l = Conv2DDNNLayer(l, name='conv2d_3x3_4', num_filters=64, filter_size=(3, 3), stride=(2, 2), **conv_kwargs)
    # 64 x 128 x 128
    l = Conv2DDNNLayer(l, name='conv2d_3x3_5', num_filters=64, filter_size=(3, 3), **conv_kwargs)
    # 64 x 128 x 128
    l = Conv2DDNNLayer(l, name='conv2d_3x3_6', num_filters=128, filter_size=(3, 3), **conv_kwargs)
    # 128 x 128 x 128
    l = MaxPool2DDNNLayer(l, name='maxpool_3x3_7', **max_pool_kwargs)
    # 128 x 64 x 64
    l = Conv2DDNNLayer(l, name='conv2d_3x3_8', num_filters=192, filter_size=(3, 3), **conv_kwargs)
    # 192 x 64 x 64
    l = Conv2DDNNLayer(l, name='conv2d_3x3_9', num_filters=192, filter_size=(3, 3), **conv_kwargs)
    # 192 x 64 x 64
    l = MaxPool2DDNNLayer(l, name='maxpool_3x3_10', **max_pool_kwargs)
    # 192 x 32 X 32

    # branch block 1
    # 192 x 32 X 32
    a_branch0 = conv2dbn(l, name='mixed_a_branch_0_1x1', num_filters=64, filter_size=(1, 1), **conv_kwargs)
    a_branch1 = conv2dbn(l, name='mixed_a_branch_1_1x1', num_filters=48, filter_size=(1, 1), **conv_kwargs)
    a_branch1 = conv2dbn(a_branch1, name='mixed_a_branch_1_5x5', num_filters=64, filter_size=(5, 5), **conv_kwargs)
    a_branch2 = conv2dbn(l, name='mixed_a_branch_2_1x1', num_filters=64, filter_size=(1, 1), **conv_kwargs)
    a_branch2 = conv2dbn(a_branch2, name='mixed_a_branch_2_3x3a', num_filters=96, filter_size=(3, 3), **conv_kwargs)
    a_branch2 = conv2dbn(a_branch2, name='mixed_a_branch_2_3x3b', num_filters=96, filter_size=(3, 3), **conv_kwargs)
    a_branch3 = Pool2DDNNLayer(l, name='mixed_a_branch_3_avgpool_3x3', **avg_pool_kwargs)
    a_branch3 = conv2dbn(a_branch3, name='mixed_a_branch_3_1x1', num_filters=32, filter_size=(1, 1), **conv_kwargs)
    a_branch_out = nn.layers.ConcatLayer([a_branch0, a_branch1, a_branch2, a_branch3], name='mixed_a_out')
    # 256 x 32 X 32

    # branch block 1
    # 256 x 32 X 32
    b_branch0 = conv2dbn(a_branch_out, name='mixed_b_branch_0_1x1', num_filters=64, filter_size=(1, 1), **conv_kwargs)
    b_branch1 = conv2dbn(a_branch_out, name='mixed_b_branch_1_1x1', num_filters=48, filter_size=(1, 1), **conv_kwargs)
    b_branch1 = conv2dbn(b_branch1, name='mixed_b_branch_1_5x5', num_filters=64, filter_size=(5, 5), **conv_kwargs)
    b_branch2 = conv2dbn(a_branch_out, name='mixed_b_branch_2_1x1', num_filters=64, filter_size=(1, 1), **conv_kwargs)
    b_branch2 = conv2dbn(b_branch2, name='mixed_b_branch_2_3x3a', num_filters=96, filter_size=(3, 3), **conv_kwargs)
    b_branch2 = conv2dbn(b_branch2, name='mixed_b_branch_2_3x3b', num_filters=96, filter_size=(3, 3), **conv_kwargs)
    b_branch3 = Pool2DDNNLayer(a_branch_out, name='mixed_b_branch_3_avgpool_3x3', **avg_pool_kwargs)
    b_branch3 = conv2dbn(b_branch3, name='mixed_b_branch_3_1x1', num_filters=64, filter_size=(1, 1), **conv_kwargs)
    b_branch_out = nn.layers.ConcatLayer([b_branch0, b_branch1, b_branch2, b_branch3], name='mixed_b_out')
    # 288 x 32 X 32

    # branch block 1
    # 288 x 32 X 32
    bc_branch0 = conv2dbn(a_branch_out, name='mixed_bc_branch_0_1x1', num_filters=64, filter_size=(1, 1), **conv_kwargs)
    bc_branch1 = conv2dbn(a_branch_out, name='mixed_bc_branch_1_1x1', num_filters=48, filter_size=(1, 1), **conv_kwargs)
    bc_branch1 = conv2dbn(bc_branch1, name='mixed_bc_branch_1_5x5', num_filters=64, filter_size=(5, 5), **conv_kwargs)
    bc_branch2 = conv2dbn(b_branch_out, name='mixed_bc_branch_2_1x1', num_filters=64, filter_size=(1, 1), **conv_kwargs)
    bc_branch2 = conv2dbn(bc_branch2, name='mixed_bc_branch_2_3x3a', num_filters=96, filter_size=(3, 3), **conv_kwargs)
    bc_branch2 = conv2dbn(bc_branch2, name='mixed_bc_branch_2_3x3b', num_filters=96, filter_size=(3, 3), **conv_kwargs)
    bc_branch3 = Pool2DDNNLayer(b_branch_out, name='mixed_bc_branch_3_avgpool_3x3', **avg_pool_kwargs)
    bc_branch3 = conv2dbn(bc_branch3, name='mixed_bc_branch_3_1x1', num_filters=64, filter_size=(1, 1), **conv_kwargs)
    bc_branch_out = nn.layers.ConcatLayer([bc_branch0, bc_branch1, bc_branch2, bc_branch3], name='mixed_bc_out')
    # 288 x 32 X 32
    b_out = nn.layers.merge.ElemwiseSumLayer([b_branch_out, bc_branch_out], name='b_bc_elemsum')
    b_out = nn.layers.NonlinearityLayer(b_out, nonlinearity=nn.nonlinearities.leaky_rectify, name='b_bc_elemsum_nl')
    b_out = nn.layers.DropoutLayer(b_out, name='b_out_drop', p=0.1)

    # branch block 2
    # 288 x 32 X 32
    c_branch0 = conv2dbn(b_out, name='mixed_c_branch_0_3x3', num_filters=384, filter_size=(3, 3), stride=2, **conv_kwargs)
    c_branch1 =  conv2dbn(b_out, name='mixed_c_branch_1_1x1', num_filters=64, filter_size=(1, 1), **conv_kwargs)
    c_branch1 = conv2dbn(c_branch1, name='mixed_c_branch_1_3x3a', num_filters=96, filter_size=(3, 3), **conv_kwargs)
    c_branch1 = conv2dbn(c_branch1, name='mixed_c_branch_1_3x3b', num_filters=96, filter_size=(3, 3), stride=2, **conv_kwargs)
    c_branch2 = MaxPool2DDNNLayer(b_out, name='mixed_c_branch_2_maxpool_3x3', **max_pool_kwargs)
    c_branch_out = nn.layers.ConcatLayer([c_branch0, c_branch1, c_branch2], name='mixed_c_out')
    # 768 x 16 X 16

    # branch block 2
    # 768 x 16 X 16
    cd1_branch0 = conv2dbn(c_branch_out, name='mixed_cd1_branch_0_1x1', num_filters=192, filter_size=(1, 1), **conv_kwargs)
    cd1_branch1 = conv2dbn(c_branch_out, name='mixed_cd1_branch_1_1x1', num_filters=128, filter_size=(1, 1), **conv_kwargs)
    cd1_branch1 = conv2dbn(cd1_branch1, name='mixed_cd1_branch_1_1x7', num_filters=128, filter_size=(1, 7), **conv_kwargs)
    cd1_branch1 = conv2dbn(cd1_branch1, name='mixed_cd1_branch_1_7x1', num_filters=192, filter_size=(7, 1), **conv_kwargs)
    cd1_branch2 = conv2dbn(c_branch_out, name='mixed_cd1_branch_2_1x1', num_filters=128, filter_size=(1, 1), **conv_kwargs)
    cd1_branch2 = conv2dbn(cd1_branch2, name='mixed_cd1_branch_2_7x1a', num_filters=128, filter_size=(7, 1), **conv_kwargs)
    cd1_branch2 = conv2dbn(cd1_branch2, name='mixed_cd1_branch_2_1x7a', num_filters=128, filter_size=(1, 7), **conv_kwargs)
    cd1_branch2 = conv2dbn(cd1_branch2, name='mixed_cd1_branch_2_7x1b', num_filters=128, filter_size=(7, 1), **conv_kwargs)
    cd1_branch2 = conv2dbn(cd1_branch2, name='mixed_cd1_branch_2_1x7b', num_filters=192, filter_size=(1, 7), **conv_kwargs)
    cd1_branch3 = Pool2DDNNLayer(c_branch_out, name='mixed_cd1_branch_3_avgpool_3x3', **avg_pool_kwargs)
    cd1_branch3 = conv2dbn(cd1_branch3, name='mixed_cd1_branch_3_1x1', num_filters=192, filter_size=(1, 1), **conv_kwargs)
    cd1_branch_out = nn.layers.ConcatLayer([cd1_branch0, cd1_branch1, cd1_branch2, cd1_branch3], name='mixed_cd1_out')
    # 768 x 16 X 16

    # branch block 2
    # 768 x 16 X 16
    cd2_branch0 = conv2dbn(c_branch_out, name='mixed_cd2_branch_0_1x1', num_filters=192, filter_size=(1, 1), **conv_kwargs)
    cd2_branch1 = conv2dbn(c_branch_out, name='mixed_cd2_branch_1_1x1', num_filters=160, filter_size=(1, 1), **conv_kwargs)
    cd2_branch1 = conv2dbn(cd2_branch1, name='mixed_cd2_branch_1_1x7', num_filters=160, filter_size=(1, 7), **conv_kwargs)
    cd2_branch1 = conv2dbn(cd2_branch1, name='mixed_cd2_branch_1_7x1', num_filters=192, filter_size=(7, 1), **conv_kwargs)
    cd2_branch2 = conv2dbn(cd1_branch_out, name='mixed_cd2_branch_2_1x1', num_filters=160, filter_size=(1, 1), **conv_kwargs)
    cd2_branch2 = conv2dbn(cd2_branch2, name='mixed_cd2_branch_2_7x1a', num_filters=160, filter_size=(7, 1), **conv_kwargs)
    cd2_branch2 = conv2dbn(cd2_branch2, name='mixed_cd2_branch_2_1x7a', num_filters=160, filter_size=(1, 7), **conv_kwargs)
    cd2_branch2 = conv2dbn(cd2_branch2, name='mixed_cd2_branch_2_7x1b', num_filters=160, filter_size=(7, 1), **conv_kwargs)
    cd2_branch2 = conv2dbn(cd2_branch2, name='mixed_cd2_branch_2_1x7b', num_filters=192, filter_size=(1, 7), **conv_kwargs)
    cd2_branch3 = Pool2DDNNLayer(cd1_branch_out, name='mixed_cd2_branch_3_avgpool_3x3', **avg_pool_kwargs)
    cd2_branch3 = conv2dbn(cd2_branch3, name='mixed_cd2_branch_3_1x1', num_filters=192, filter_size=(1, 1), **conv_kwargs)
    cd2_branch_out = nn.layers.ConcatLayer([cd2_branch0, cd2_branch1, cd2_branch2, cd2_branch3], name='mixed_cd2_out')
    # 768 x 16 X 16

    # branch block 2
    # 768 x 16 X 16
    cd3_branch0 = conv2dbn(cd1_branch_out, name='mixed_cd3_branch_0_1x1', num_filters=192, filter_size=(1, 1), **conv_kwargs)
    cd3_branch1 = conv2dbn(cd1_branch_out, name='mixed_cd3_branch_1_1x1', num_filters=160, filter_size=(1, 1), **conv_kwargs)
    cd3_branch1 = conv2dbn(cd3_branch1, name='mixed_cd3_branch_1_1x7', num_filters=160, filter_size=(1, 7), **conv_kwargs)
    cd3_branch1 = conv2dbn(cd3_branch1, name='mixed_cd3_branch_1_7x1', num_filters=192, filter_size=(7, 1), **conv_kwargs)
    cd3_branch2 = conv2dbn(cd2_branch_out, name='mixed_cd3_branch_2_1x1', num_filters=160, filter_size=(1, 1), **conv_kwargs)
    cd3_branch2 = conv2dbn(cd3_branch2, name='mixed_cd3_branch_2_7x1a', num_filters=160, filter_size=(7, 1), **conv_kwargs)
    cd3_branch2 = conv2dbn(cd3_branch2, name='mixed_cd3_branch_2_1x7a', num_filters=160, filter_size=(1, 7), **conv_kwargs)
    cd3_branch2 = conv2dbn(cd3_branch2, name='mixed_cd3_branch_2_7x1b', num_filters=160, filter_size=(7, 1), **conv_kwargs)
    cd3_branch2 = conv2dbn(cd3_branch2, name='mixed_cd3_branch_2_1x7b', num_filters=192, filter_size=(1, 7), **conv_kwargs)
    cd3_branch3 = Pool2DDNNLayer(cd2_branch_out, name='mixed_cd3_branch_3_avgpool_3x3', **avg_pool_kwargs)
    cd3_branch3 = conv2dbn(cd3_branch3, name='mixed_cd3_branch_3_1x1', num_filters=192, filter_size=(1, 1), **conv_kwargs)
    cd3_branch_out = nn.layers.ConcatLayer([cd3_branch0, cd3_branch1, cd3_branch2, cd3_branch3], name='mixed_cd3_out')
    # 768 x 16 X 16

    # branch block 2
    # 768 x 16 X 16
    cd4_branch0 = conv2dbn(c_branch_out, name='mixed_cd4_branch_0_1x1', num_filters=192, filter_size=(1, 1), **conv_kwargs)
    cd4_branch1 = conv2dbn(cd1_branch_out, name='mixed_cd4_branch_1_1x1', num_filters=192, filter_size=(1, 1), **conv_kwargs)
    cd4_branch1 = conv2dbn(cd4_branch1, name='mixed_cd4_branch_1_1x7', num_filters=192, filter_size=(1, 7), **conv_kwargs)
    cd4_branch1 = conv2dbn(cd4_branch1, name='mixed_cd4_branch_1_7x1', num_filters=192, filter_size=(7, 1), **conv_kwargs)
    cd4_branch2 = conv2dbn(cd2_branch_out, name='mixed_cd4_branch_2_1x1', num_filters=192, filter_size=(1, 1), **conv_kwargs)
    cd4_branch2 = conv2dbn(cd4_branch2, name='mixed_cd4_branch_2_7x1a', num_filters=192, filter_size=(7, 1), **conv_kwargs)
    cd4_branch2 = conv2dbn(cd4_branch2, name='mixed_cd4_branch_2_1x7a', num_filters=192, filter_size=(1, 7), **conv_kwargs)
    cd4_branch2 = conv2dbn(cd4_branch2, name='mixed_cd4_branch_2_7x1b', num_filters=192, filter_size=(7, 1), **conv_kwargs)
    cd4_branch2 = conv2dbn(cd4_branch2, name='mixed_cd4_branch_2_1x7b', num_filters=192, filter_size=(1, 7), **conv_kwargs)
    cd4_branch3 = Pool2DDNNLayer(cd3_branch_out, name='mixed_cd4_branch_3_avgpool_3x3', **avg_pool_kwargs)
    cd4_branch3 = conv2dbn(cd4_branch3, name='mixed_cd4_branch_3_1x1', num_filters=192, filter_size=(1, 1), **conv_kwargs)
    cd4_branch_out = nn.layers.ConcatLayer([cd4_branch0, cd4_branch1, cd4_branch2, cd4_branch3], name='mixed_cd4_out')
    # 768 x 16 X 16
    c_out = nn.layers.merge.ElemwiseSumLayer([c_branch_out, cd4_branch_out], name='c_cd4_elemsum')
    c_out = nn.layers.NonlinearityLayer(c_out, nonlinearity=nn.nonlinearities.leaky_rectify, name='c_cd4_elemsum_nl')
    c_out = nn.layers.DropoutLayer(c_out, name='c_out_drop', p=0.2)

    # branch block 3
    # 768 x 16 X 16
    d_branch0 = conv2dbn(c_out, name='mixed_d_branch_0_1x1', num_filters=192, filter_size=(1, 1), **conv_kwargs)
    d_branch0 = conv2dbn(d_branch0, name='mixed_d_branch_0_3x3', num_filters=320, filter_size=(3, 3), stride=2, **conv_kwargs)
    d_branch1 = conv2dbn(c_out, name='mixed_d_branch_1_1x1', num_filters=192, filter_size=(1, 1), **conv_kwargs)
    d_branch1 = conv2dbn(d_branch1, name='mixed_d_branch_1_1x7', num_filters=192, filter_size=(1, 7), **conv_kwargs)
    d_branch1 = conv2dbn(d_branch1, name='mixed_d_branch_1_7x1', num_filters=192, filter_size=(7, 1), **conv_kwargs)
    d_branch1 = conv2dbn(d_branch1, name='mixed_d_branch_1_3x3', num_filters=192, filter_size=(3, 3), stride=2, **conv_kwargs)
    d_branch2 = MaxPool2DDNNLayer(c_out, name='mixed_d_branch_2_maxpool_1x1', **max_pool_kwargs)
    d_branch_out = nn.layers.ConcatLayer([d_branch0, d_branch1, d_branch2], name='mixed_d_out')
    # 1280 x 8 X 8

    # branch block 3
    # 1280 x 8 X 8
    e_branch0 = conv2dbn(d_branch_out, name='mixed_e_branch_0_1x1', num_filters=320, filter_size=(1, 1), **conv_kwargs)
    e_branch1 = conv2dbn(d_branch_out, name='mixed_e_branch_1_1x1', num_filters=384, filter_size=(1, 1), **conv_kwargs)
    e_branch1_1 = conv2dbn(e_branch1, name='mixed_e_branch_1_1x3', num_filters=384, filter_size=(1, 3), **conv_kwargs)
    e_branch1_2 = conv2dbn(e_branch1, name='mixed_e_branch_1_3x1', num_filters=384, filter_size=(3, 1), **conv_kwargs)
    e_branch1_concat = nn.layers.ConcatLayer([e_branch1_1, e_branch1_2], name='mixed_e_branch_1_concat')
    e_branch2 = conv2dbn(d_branch_out, name='mixed_e_branch_2_1x1', num_filters=448, filter_size=(1, 1), **conv_kwargs)
    e_branch2 = conv2dbn(e_branch2, name='mixed_e_branch_2_3x3', num_filters=384, filter_size=(3, 3), **conv_kwargs)
    e_branch2_1 = conv2dbn(e_branch2, name='mixed_e_branch_2_1x3', num_filters=384, filter_size=(1, 3), **conv_kwargs)
    e_branch2_2 = conv2dbn(e_branch2, name='mixed_e_branch_2_3x1', num_filters=384, filter_size=(3, 1), **conv_kwargs)
    e_branch2_concat = nn.layers.ConcatLayer([e_branch2_1, e_branch2_2], name='mixed_e_branch_2_concat')
    e_branch3 = Pool2DDNNLayer(d_branch_out, name='mixed_e_branch_3_avgpool_3x3', **avg_pool_kwargs)
    e_branch3 = conv2dbn(e_branch3, name='mixed_e_branch_3_1x1', num_filters=192, filter_size=(1, 1), **conv_kwargs)
    e_branch_out = nn.layers.ConcatLayer([e_branch0, e_branch1_concat, e_branch2_concat, e_branch3], name='mixed_e_out')
    # 2048 x 8 X 8

    # branch block 3
    # 2048 x 8 X 8
    eo_branch0 = conv2dbn(d_branch_out, name='mixed_eo_branch_0_1x1', num_filters=320, filter_size=(1, 1), **conv_kwargs)
    eo_branch1 = conv2dbn(d_branch_out, name='mixed_eo_branch_1_1x1', num_filters=384, filter_size=(1, 1), **conv_kwargs)
    eo_branch1_1 = conv2dbn(eo_branch1, name='mixed_eo_branch_1_1x3', num_filters=384, filter_size=(1, 3), **conv_kwargs)
    eo_branch1_2 = conv2dbn(eo_branch1, name='mixed_eo_branch_1_3x1', num_filters=384, filter_size=(3, 1), **conv_kwargs)
    eo_branch1_concat = nn.layers.ConcatLayer([eo_branch1_1, eo_branch1_2], name='mixed_eo_branch_1_concat')
    eo_branch2 = conv2dbn(e_branch_out, name='mixed_eo_branch_2_1x1', num_filters=448, filter_size=(1, 1), **conv_kwargs)
    eo_branch2 = conv2dbn(eo_branch2, name='mixed_eo_branch_2_3x3', num_filters=384, filter_size=(3, 3), **conv_kwargs)
    eo_branch2_1 = conv2dbn(eo_branch2, name='mixed_eo_branch_2_1x3', num_filters=384, filter_size=(1, 3), **conv_kwargs)
    eo_branch2_2 = conv2dbn(eo_branch2, name='mixed_eo_branch_2_3x1', num_filters=384, filter_size=(3, 1), **conv_kwargs)
    eo_branch2_concat = nn.layers.ConcatLayer([eo_branch2_1, eo_branch2_2], name='mixed_eo_branch_2_concat')
    eo_branch3 = Pool2DDNNLayer(e_branch_out, name='mixed_eo_branch_3_avgpool_3x3', **avg_pool_kwargs)
    eo_branch3 = conv2dbn(eo_branch3, name='mixed_eo_branch_3_1x1', num_filters=192, filter_size=(1, 1), **conv_kwargs)
    eo_branch_out = nn.layers.ConcatLayer([eo_branch0, eo_branch1_concat, eo_branch2_concat, eo_branch3], name='mixed_eo_out')
    # 2048 x 8 X 8
    e_out = nn.layers.merge.ElemwiseSumLayer([e_branch_out, eo_branch_out], name='e_eo_elemsum')
    e_out = nn.layers.NonlinearityLayer(e_out, nonlinearity=nn.nonlinearities.leaky_rectify, name='e_eo_elemsum_nl')
    e_out = nn.layers.DropoutLayer(e_out, name='e_out_drop', p=0.3)

    # output
    # 2048 x 8 X 8
    out = Pool2DDNNLayer(e_out, name='out_avgpool', pool_size=8, stride=1, mode='average_exc_pad')
    # 2048 x 1 X 1
    if regression:
        out = nn.layers.DropoutLayer(out, name='out_drop', p=0.5)
        # 2048 x 1 x 1
        out = nn.layers.DenseLayer(out, name='out_dense', num_units=1024, **dense_kwargs)
        # 1024
        out = nn.layers.FeaturePoolLayer(out, name='out_featurepool', pool_size=2)
        # 512
        out = nn.layers.DenseLayer(out, name='out', num_units=output_num)
    else:
        out = Conv2DDNNLayer(out, name='out_conv', num_filters=output_num, filter_size=(1, 1), nonlinearity=None)
        out = nn.layers.FlattenLayer(out, name='out')
        out = nn.layers.NonlinearityLayer(out, name='softmax', nonlinearity=nn.nonlinearities.softmax)

    return out
