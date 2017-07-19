import paramiko
import os


def ssh_connect_pwd(_host, _username, _password=None):
    try:
        _ssh_fd = paramiko.SSHClient()
        _ssh_fd.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        _ssh_fd.connect(_host, username = _username, password = _password)
    except Exception as e:
        print('ssh %s@%s: %s' % (_username, _host, e))
        exit()
    return _ssh_fd


def ssh_connect_rsa(_host, _username, key_pwd=None):
    try:
        _ssh_fd = paramiko.SSHClient()
        _ssh_fd.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        privatekeyfile = os.path.expanduser('~/.ssh/id_rsa')
        mykey = paramiko.RSAKey.from_private_key_file(privatekeyfile, password=key_pwd)
        _ssh_fd.connect(_host, username = _username, pkey = mykey)
    except Exception as e:
        print('ssh %s@%s: %s' % (_username, _host, e))
        exit()
    return _ssh_fd


def sftp_open(_ssh_fd):
    return _ssh_fd.open_sftp()


def sftp_put(_sftp_fd, _put_from_path, _put_to_path):
    return _sftp_fd.put(_put_from_path, _put_to_path)


def sftp_get(_sftp_fd, _get_from_path, _get_to_path):
    return _sftp_fd.get(_get_from_path, _get_to_path)


def sftp_close(_sftp_fd):
    _sftp_fd.close()


def ssh_close(_ssh_fd):
    _ssh_fd.close()


if __name__ == '__main__':
    sshd = ssh_connect_rsa('192.168.20.10', 'vagrant')
    sftpd = sftp_open(sshd)
    try:
        sftp_get(sftpd, 'test.txt', '/home/terrypang/test.txt')
    except Exception as e:
        print('ERROR: sftp_get - %s' % e)
    sftp_close(sftpd)
    ssh_close(sshd)
