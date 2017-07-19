from __future__ import division
from __future__ import print_function

import pandas as pd
import click
import os
from glob import glob
from PIL import Image
from sklearn import cross_validation
import numpy as np
from tqdm import tqdm
import shutil
import config
from sklearn.utils.class_weight import compute_class_weight


RANDOM_STATE = 9


def square_bbox(img):
    w, h = img.size
    if w < h:
        left = 0
        upper = (h - w) // 2
        right = w
        lower = upper + w
    elif w > h:
        left = (w - h) // 2
        upper = 0
        right = left + h
        lower = h
    else:
        left = 0
        upper = 0
        right = w
        lower = h
    return (left, upper, right, lower)


def convert(srcFile, desFile, crop_size):
    img = Image.open(srcFile)
    bbox = square_bbox(img)
    cropped = img.crop(bbox)
    resized = cropped.resize([crop_size, crop_size])
    resized.save(desFile)


def genConvert(crop_size):
    raw_path = 'data/raw'
    convert_path = 'data/convert_' + str(crop_size)
    files = glob('{}/*'.format(raw_path))
    try:
        os.mkdir(convert_path)
    except OSError:
        pass
    index = 1
    for srcFile in tqdm(files):
        desFile = srcFile.replace(raw_path, convert_path).replace('jpeg', 'jpg')
        convert(srcFile, desFile, crop_size)
        index += 1
    print('done')


def split(crop_size, split_size):
    data = pd.read_csv('data/trainLabels.csv', index_col=0)
    convert_path = 'data/convert_' + str(crop_size)
    train_path = 'data/train_' + str(crop_size)
    validation_path = 'data/validation_' + str(crop_size)
    try:
        os.mkdir(train_path)
        os.mkdir(train_path + '/0')
        os.mkdir(train_path + '/1')
        os.mkdir(train_path + '/2')
        os.mkdir(train_path + '/3')
        os.mkdir(train_path + '/4')
        os.mkdir(validation_path)
        os.mkdir(validation_path + '/0')
        os.mkdir(validation_path + '/1')
        os.mkdir(validation_path + '/2')
        os.mkdir(validation_path + '/3')
        os.mkdir(validation_path + '/4')
    except OSError:
        pass
    files = sorted(glob('{}/*'.format(convert_path)))
    names = [os.path.basename(x).split('.')[0] for x in files]
    labels = pd.read_csv('data/trainLabels.csv', index_col=0).loc[names].values.flatten()
    left = np.array(['left' in n for n in names])
    left_right_labels = np.vstack([labels[left], labels[~left]]).T
    spl = cross_validation.StratifiedShuffleSplit(left_right_labels[:, 0],
                                                  test_size=split_size,
                                                  random_state=RANDOM_STATE,
                                                  n_iter=1)
    tr, te = next(iter(spl))
    train_index = np.hstack([tr * 2, tr * 2 + 1])
    validation_index = np.hstack([te * 2, te * 2 + 1])

    y = []
    dr0_num = 0
    dr1_num = 0
    dr2_num = 0
    dr3_num = 0
    dr4_num = 0
    for i in tqdm(train_index):
        srcFile = files[i]
        name = os.path.basename(srcFile).split('.')[0]
        label = data.loc[name].values[0]
        if label == 0:
            y.append(0)
            desPath = train_path + '/0'
            dr0_num += 1
        elif label == 1:
            y.append(1)
            desPath = train_path + '/1'
            dr1_num += 1
        elif label == 2:
            y.append(2)
            desPath = train_path + '/2'
            dr2_num += 1
        elif label == 3:
            y.append(3)
            desPath = train_path + '/3'
            dr3_num += 1
        elif label == 4:
            y.append(4)
            desPath = train_path + '/4'
            dr4_num += 1
        shutil.copy(srcFile, desPath)
    print('train_dr0: {}'.format(dr0_num))
    print('train_dr1: {}'.format(dr1_num))
    print('train_dr2: {}'.format(dr2_num))
    print('train_dr3: {}'.format(dr3_num))
    print('train_dr4: {}'.format(dr4_num))

    dr0_num = 0
    dr1_num = 0
    dr2_num = 0
    dr3_num = 0
    dr4_num = 0
    for i in tqdm(validation_index):
        srcFile = files[i]
        name = os.path.basename(srcFile).split('.')[0]
        label = data.loc[name].values[0]
        if label == 0:
            desPath = validation_path + '/0'
            dr0_num += 1
        elif label == 1:
            desPath = validation_path + '/1'
            dr1_num += 1
        elif label == 2:
            desPath = validation_path + '/2'
            dr2_num += 1
        elif label == 3:
            desPath = validation_path + '/3'
            dr3_num += 1
        elif label == 4:
            desPath = validation_path + '/4'
            dr4_num += 1
        shutil.copy(srcFile, desPath)
    print('validation_dr0: {}'.format(dr0_num))
    print('validation_dr1: {}'.format(dr1_num))
    print('validation_dr2: {}'.format(dr2_num))
    print('validation_dr3: {}'.format(dr3_num))
    print('validation_dr4: {}'.format(dr4_num))

    cw = compute_class_weight("balanced", [0, 1, 2, 3, 4], y)
    print('train: {}'.format(train_index.shape))
    print('validation: {}'.format(validation_index.shape))
    print('class_weight: {}'.format(cw))


@click.command()
@click.option('--generate', is_flag=True, default=False, show_default=True,
              help="Generate convert images")
def main(generate):
    if generate:
        genConvert(config.crop_size)
    split(config.crop_size, config.split_size)


if __name__ == '__main__':
    main()
