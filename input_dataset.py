import os
from os.path import isfile, isdir
from six.moves import xrange
from tqdm import tqdm
import tarfile
import tensorflow as tf
from urllib.request import urlretrieve

from My_cifar10_image_classification import helper


def cifar10_input():
    """Download and extract the tarball from toronto's website."""
    cifar10_dataset_folder_path = 'cifar-10-batches-py'
    tar_gz_path = 'cifar-10-python.tar.gz'
    preprocess_batch_folder_path = 'preprocess_batch'

    class DLProgress(tqdm):
        last_block = 0

        def hook(self, block_num=1, block_size=1, total_size=None):
            self.total = total_size
            self.update((block_num - self.last_block) * block_size)
            self.last_block = block_num

    if not isfile(tar_gz_path):
        with DLProgress(unit='B', unit_scale=True, miniters=1, desc='CIFAR-10 Dataset') as pbar:
            urlretrieve(
                'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
                tar_gz_path,
                pbar.hook)

    if not isdir(cifar10_dataset_folder_path):
        with tarfile.open(tar_gz_path) as tar:
            tar.extractall()
            tar.close()

    files = os.listdir(os.getcwd() + '/preprocess_batch')
    if len(files) == 0:
        helper.preprocess_and_save_data(cifar10_dataset_folder_path)
        print('All data is preprocessed!')

    filenames = [os.path.join(preprocess_batch_folder_path, 'preprocess_batch_%d.p' % i)
                 for i in xrange(1, 6)]
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)