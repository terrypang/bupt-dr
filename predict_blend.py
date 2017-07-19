import numpy as np
import theano
import config
from nolearn.lasagne import BatchIterator
from utils import iterator, util, sftp, tta, metrics, augmentation
from utils.nolearn_net import NeuralNet
import time
from redis import Redis
import pymysql.cursors
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from glob import glob
import os
import pandas as pd
from sklearn.metrics import confusion_matrix


def get_blend_net(weight):
    model = util.load_module(config.blend_model)
    regression = True
    output_num = 1

    args = {
        'layers': model.build_model(config.blend_depth, config.blend_size, output_num, regression),
        'batch_iterator_train': BatchIterator(batch_size=128),
        'batch_iterator_test': BatchIterator(batch_size=128),
        'regression': regression,
        'update_learning_rate': theano.shared(np.cast['float32'](0.01)),
    }
    net = NeuralNet(**args)

    print("Model is {}, and objective is {}".format(config.blend_model, 'regression'))
    try:
        net.load_params_from(weight)
        print("loaded weights from {}".format(weight))
    except IOError:
        print("couldn't load weights starting from scratch")
        raise StopIteration()

    return net


def transform(net, fileList, n_iter, skip, scaler):
    tfs, color_vecs = tta.build_quasirandom_transforms(n_iter, skip=skip, color_sigma=config.sigma, **config.aug_params)
    Xs, Xs2 = None, None
    for i, (tf, color_vec) in enumerate(zip(tfs, color_vecs), start=1):
        X = net.transform(fileList, config.output_layer, transform=tf, color_vec=color_vec)
        if Xs is None:
            Xs = X
            Xs2 = X ** 2
        else:
            Xs += X
            Xs2 += X ** 2

    mean = Xs / n_iter
    std = np.sqrt((Xs2 - Xs ** 2 / n_iter) / (n_iter - 1))
    data = [mean, std]
    data = [X.reshape([X.shape[0], -1]) for X in data]
    data = np.hstack(data)
    data = scaler.transform(data)
    n = len(data)
    left_idx = np.arange(n)
    right_idx = left_idx + np.sign(2 * ((left_idx + 1) % 2) - 1)
    left_right_data = np.hstack([data[left_idx], data[right_idx]]).astype(np.float32)
    return left_right_data


def test():
    model = util.load_module(config.model)

    if config.objective == 'classification':
        regression = False
        output_num = 5
    elif config.objective == 'regression':
        regression = True
        output_num = 1
    elif config.objective == 'kappa':
        regression = True
        output_num = 5
    else:
        raise TypeError('objective type error')

    weights_from = config.predict_weight

    args = {
        'layers': model.build_model(config.img_width, config.img_height, output_num, regression),
        'batch_iterator_train': iterator.ResampleIterator(batch_size=config.batch_size_train),
        'batch_iterator_test': iterator.SharedIterator(deterministic=True, batch_size=config.batch_size_test),
        'regression': regression,
        'update_learning_rate': theano.shared(np.cast['float32'](config.schedule[0])),
    }
    feature_net = NeuralNet(**args)

    print("Model is {}, and objective is {}".format(config.model, config.objective))
    try:
        feature_net.load_params_from(weights_from)
        print("loaded weights from {}".format(weights_from))
    except IOError:
        print("couldn't load weights starting from scratch")
        raise StopIteration()

    blend_nets = []
    for weight in config.blend_weights:
        net = get_blend_net(weight)
        blend_nets.append(net)

    file_path = 'data/convert_1024'
    fs = sorted(glob('{}/*'.format(file_path)))
    names = [os.path.basename(x).split('.')[0] for x in fs]
    labels = pd.read_csv(config.label_file, index_col=0).loc[names].values.flatten()
    labels = np.array(labels, dtype=np.float32)

    data = []
    result = []
    scalers = []

    for files in config.feature_files:
        feature = [np.load(f) for f in files]
        feature = [X.reshape([X.shape[0], -1]) for X in feature]
        feature = np.hstack(feature)
        scaler = StandardScaler().fit(feature)
        scalers.append(scaler)

    tic = time.time()
    data.append(transform(feature_net, fs, 50, 0, scalers[0]))
    print('took {:6.1f} seconds'.format(time.time() - tic))
    data.append(transform(feature_net, fs, 50, 50, scalers[1]))
    print('took {:6.1f} seconds'.format(time.time() - tic))
    data.append(transform(feature_net, fs, 50, 100, scalers[2]))
    print('took {:6.1f} seconds'.format(time.time() - tic))

    i = 0
    for blend_net in blend_nets:
        X = data[i].reshape(data[i].shape[0], config.blend_depth, config.blend_size, config.blend_size)
        y_pred = blend_net.predict(X)
        print(y_pred)
        result.append(y_pred)
        i += 1;

    result = np.mean(result, axis=0)
    result[np.isnan(result)] = 0
    result = np.clip(result, 0, 4)
    result = np.round(result).astype(int).ravel()
    print(result)
    result = result.reshape(-1, 1)
    print("Blend final accuracy score: {}".format(metrics.accuracy(labels, result)))
    print("Blend final kappa score: {}".format(metrics.kappa(labels, result)))
    print("confusion matrix")
    print(confusion_matrix(labels, result, labels=[0, 1, 2, 3, 4]))


def main():
    model = util.load_module(config.model)

    if config.objective == 'classification':
        regression = False
        output_num = 5
    elif config.objective == 'regression':
        regression = True
        output_num = 1
    elif config.objective == 'kappa':
        regression = True
        output_num = 5
    else:
        raise TypeError('objective type error')

    weights_from = config.predict_weight

    args = {
        'layers': model.build_model(config.img_width, config.img_height, output_num, regression),
        'batch_iterator_train': iterator.ResampleIterator(batch_size=config.batch_size_train),
        'batch_iterator_test': iterator.SharedIterator(deterministic=True, batch_size=config.batch_size_test),
        'regression': regression,
        'update_learning_rate': theano.shared(np.cast['float32'](config.schedule[0])),
    }
    feature_net = NeuralNet(**args)

    print("Model is {}, and objective is {}".format(config.model, config.objective))
    try:
        feature_net.load_params_from(weights_from)
        print("loaded weights from {}".format(weights_from))
    except IOError:
        print("couldn't load weights starting from scratch")
        raise StopIteration()

    blend_nets = []
    for weight in config.blend_weights:
        net = get_blend_net(weight)
        blend_nets.append(net)

    scalers = []
    for files in config.feature_files:
        feature = [np.load(f) for f in files]
        feature = [X.reshape([X.shape[0], -1]) for X in feature]
        feature = np.hstack(feature)
        scaler = StandardScaler().fit(feature)
        scalers.append(scaler)

    print("predicting ...")
    redis = Redis(host=config.remote_ip, port=6973, password='password')

    while True:
        num = redis.llen('task_queue')
        if num == 0:
            time.sleep(1)
        else:
            fs = []
            task_ids = []
            for i in range(num):
                res = redis.rpop('task_queue')
                task_id = res.decode()
                task_ids.append(task_id)
                remote_left_path = '/var/www/jykj-demo/storage/app/' + task_id + '_left.jpg'
                local_left_path = '/home/terrypang/Desktop/jingyu-dr/data/test/' + task_id + '_left.jpg'
                remote_right_path = '/var/www/jykj-demo/storage/app/' + task_id + '_right.jpg'
                local_right_path = '/home/terrypang/Desktop/jingyu-dr/data/test/' + task_id + '_right.jpg'

                sshd = sftp.ssh_connect_pwd(config.remote_ip, 'username', 'password')
                sftpd = sftp.sftp_open(sshd)
                try:
                    sftp.sftp_get(sftpd, remote_left_path, local_left_path)
                    sftp.sftp_get(sftpd, remote_right_path, local_right_path)
                except Exception as e:
                    print('ERROR: sftp_get - %s' % e)
                sftp.sftp_close(sftpd)
                sftp.ssh_close(sshd)
                fs.append(local_left_path)
                fs.append(local_right_path)

            print(datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
            print(fs)

            data = []
            results = []
            tic = time.time()
            data.append(transform(feature_net, fs, 50, 0, scalers[0]))
            print('took {:6.1f} seconds'.format(time.time() - tic))
            data.append(transform(feature_net, fs, 50, 50, scalers[1]))
            print('took {:6.1f} seconds'.format(time.time() - tic))
            data.append(transform(feature_net, fs, 50, 100, scalers[2]))
            print('took {:6.1f} seconds'.format(time.time() - tic))

            i = 0
            for blend_net in blend_nets:
                print(data[i])
                print(data[i].shape)
                X = data[i].reshape(data[i].shape[0], config.blend_depth, config.blend_size, config.blend_size)
                y_pred = blend_net.predict(X)
                results.append(y_pred)
                i+=1;

            print(results)
            results = np.mean(results, axis=0)
            results = results.reshape(-1, 2)
            print(results)
            for i in range(len(results)):
                result = results[i]
                print(result)
                result_left = util.getResult(result[0])
                result_right = util.getResult(result[1])
                print(result_left)
                print(result_right)
                connection = pymysql.connect(host=config.remote_ip,
                                             user='homestead',
                                             password='secret',
                                             db='homestead',
                                             charset='utf8mb4',
                                             cursorclass=pymysql.cursors.DictCursor)
                try:
                    with connection.cursor() as cursor:
                        # Create a new record
                        sql = "INSERT INTO `tasks` (`task_id`, `left`, `right`) VALUES (%s, %s, %s)"
                        cursor.execute(sql, (task_ids[i], result_left, result_right))

                    # connection is not autocommit by default. So you must commit to save
                    # your changes.
                    connection.commit()
                finally:
                    connection.close()

            print(datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
            print("predicting ...")


if __name__ == '__main__':
    main()
    # test()
