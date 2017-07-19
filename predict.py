import numpy as np
import theano
import config
from utils import iterator, util, sftp
from utils.nolearn_net import NeuralNet
import time
from redis import Redis
import pymysql.cursors
from datetime import datetime


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
    net = NeuralNet(**args)

    print("Model is {}, and objective is {}".format(config.model, config.objective))
    try:
        net.load_params_from(weights_from)
        print("loaded weights from {}".format(weights_from))
    except IOError:
        print("couldn't load weights starting from scratch")
        raise StopIteration()

    print("predicting ...")
    redis = Redis(host=config.remote_ip, port=6973, password='zenu200404')

    while True:
        res = redis.rpop('file_queue')
        if res == None:
            time.sleep(1)
        else:
            filename = res.decode()
            remote_path = '/var/www/jykj-demo/storage/app/' + filename
            local_path = '/home/terrypang/Desktop/jingyu-dr/data/test/q_' + filename

            sshd = sftp.ssh_connect_pwd(config.remote_ip, 'ubuntu', 'zenu200404')
            sftpd = sftp.sftp_open(sshd)
            try:
                sftp.sftp_get(sftpd, remote_path, local_path)
            except Exception as e:
                print('ERROR: sftp_get - %s' % e)
            sftp.sftp_close(sftpd)
            sftp.ssh_close(sshd)

            print(datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
            fs = [local_path]
            print(fs)
            y_pred = net.predict(fs)
            print(y_pred)
            result = util.getResult(y_pred)
            connection = pymysql.connect(host=config.remote_ip,
                                         user='homestead',
                                         password='secret',
                                         db='homestead',
                                         charset='utf8mb4',
                                         cursorclass=pymysql.cursors.DictCursor)
            try:
                with connection.cursor() as cursor:
                    # Create a new record
                    sql = "INSERT INTO `files` (`file`, `result`) VALUES (%s, %s)"
                    cursor.execute(sql, (filename, result))

                # connection is not autocommit by default. So you must commit to save
                # your changes.
                connection.commit()
            finally:
                connection.close()

            print(datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
            print("predicting ...")


if __name__ == '__main__':
    main()
