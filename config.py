import numpy as np

# convert settings
crop_size = 1024
split_size = 0.1

# channel standard deviations
STD = np.array([81.97618866, 53.23217773, 42.93052292], dtype=np.float32)

# channel means
MEAN = np.array([41.16128922, 41.05914307, 29.54789543], dtype=np.float32)

# for color augmentation, computed with make_pca.py
U = np.array([[-0.58456488, 0.71514759, 0.38173861],
              [-0.58904609, -0.0508885, -0.80589375],
              [-0.55771769, -0.69611413, 0.45109312]], dtype=np.float32)
EV = np.array([2.00837354, 0.48712317, 0.16402843], dtype=np.float32)

# for class balance
balance = {
    'class_weights': np.array([0.2716686, 2.90399633, 1.33234984, 8.03303685, 10.03492063], dtype=np.float32),
    'balance_ratio': 0.975,
    'final_balance_weights': np.array([1, 2, 2, 2, 2], dtype=np.float32),
}

# data settings
label_file = 'data/trainLabels.csv'
raw_width, raw_height = 580, 580
img_width, img_height = 512, 512

# data augmentation
aug_params = {
    'zoom_range': (1 / 1.15, 1.15),
    'rotation_range': (0, 360),
    'shear_range': (0, 10),
    'translation_range': (-40, 40),
    'do_flip': True,
    'allow_stretch': True,
}
sigma = 0.5
no_augmentation_params = {
    'zoom_range': (1.0, 1.0),
    'rotation_range': (0, 0),
    'shear_range': (0, 0),
    'translation_range': (0, 0),
    'do_flip': False,
    'allow_stretch': False,
}

# network
model = 'models/drnet.py'
batch_size_train = 32
batch_size_test = 32
schedule = {
    0: 0.003,
    100: 0.001,
    150: 0.0003,
    200: 0.0001,
    250: 0.00003,
    300: 0.00001,
    351: 'stop',
}

# transform
feature_path = 'data/features'
output_layer = 'out_avgpool'
objective = 'classification'
transform_weight = 'weights/drnet/weights_final_classification_2016-11-27-02-24-19.pkl'

# blend
blend_model = 'models/drnet_blend.py'
blend_size = 32
blend_depth = 8

# predict
predict_weight = 'weights/drnet/weights_final_classification_2016-11-27-02-24-19.pkl'
blend_weights = ['weights/drnet/weights_final_regression_blend_classification_features_50_0.pkl',
                    'weights/drnet/weights_final_regression_blend_classification_features_50_50.pkl',
                    'weights/drnet/weights_final_regression_blend_classification_features_50_100.pkl']
feature_files = [
    ['data/features/drnet_classification_mean_iter_50_skip_0.npy', 'data/features/drnet_classification_std_iter_50_skip_0.npy'],
    ['data/features/drnet_classification_mean_iter_50_skip_50.npy', 'data/features/drnet_classification_std_iter_50_skip_50.npy'],
    ['data/features/drnet_classification_mean_iter_50_skip_100.npy', 'data/features/drnet_classification_std_iter_50_skip_100.npy'],
]
remote_ip = '139.199.108.142'
