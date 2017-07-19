from __future__ import division, print_function

import numpy as np
from glob import glob
from utils import augmentation
import config
from tqdm import tqdm


def main():

    filenames = glob('{}/*'.format('data/convert_' + str(config.crop_size)))

    bs = 100
    batches = [filenames[i * bs : (i + 1) * bs]
               for i in range(int(len(filenames) / bs) + 1)]

    STDs, MEANs = [], []
    Us, EVs = [], []
    for batch in tqdm(batches):
        images = np.array([augmentation.load_image(f, config.crop_size, config.crop_size) for f in batch])
        X = images.transpose(0, 2, 3, 1).reshape(-1, 3)
        STD = np.std(X, axis=0)
        MEAN = np.mean(X, axis=0)
        STDs.append(STD)
        MEANs.append(MEAN)

        X = np.subtract(X, MEAN)
        X = np.divide(X, STD)
        cov = np.dot(X.T, X) / X.shape[0]
        U, S, V = np.linalg.svd(cov)
        ev = np.sqrt(S)
        Us.append(U)
        EVs.append(ev)

    print('STD')
    print(np.mean(STDs, axis=0))
    print('MEAN')
    print(np.mean(MEANs, axis=0))
    print('U')
    print(np.mean(Us, axis=0))
    print('EV')
    print(np.mean(EVs, axis=0))

if __name__ == '__main__':
    main()
