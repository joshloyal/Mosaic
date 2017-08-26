import os
import urllib.request as url_request
import struct
import gzip

import numpy as np

from mosaic.datasets.base import get_data_home
from mosaic.datasets.progress_bar import chunk_read


HERE = os.path.dirname(os.path.abspath(__file__))
URL = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
DATA_NAME = "fashion"

TRAIN_DATA = dict(description='training set images',
                  filename='train-images-idx3-ubyte.gz')
TRAIN_LABELS = dict(description='training set labels',
                    filename='train-labels-idx1-ubyte.gz')
TEST_DATA = dict(description='test set images',
                 filename='t10k-images-idx3-ubyte.gz')
TEST_LABELS = dict(description='test set labels',
                   filename='t10k-labels-idx1-ubyte.gz')


def download_fashion_mnist(target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for datum in (TRAIN_DATA, TRAIN_LABELS, TEST_DATA, TEST_LABELS):
        archive_path = os.path.join(target_dir, datum['filename'])
        print(80 * '-')
        print("Downloading fashion-mnist {description} to {filename}".format(
            description=datum['description'],
            filename=archive_path))
        opener = url_request.urlopen(URL + datum['filename'])
        with open(archive_path, 'wb') as f:
            f.write(chunk_read(opener))


def load_fashion(path, kind='train'):
    """Loading utility modified from
       https://github.com/zalandoresearch/fashion-mnist/blob/master/utils/mnist_reader.py
    """
    if kind not in ['t10k', 'train', 'test']:
        raise ValueError("Unrecognized dataset partition. `kind` = {kind}, but must be "
                         "{'train', 'test', 't10k'}".format(kind=kind))

    if kind == 'test':
        kind = 't10k'

    labels_path = os.path.join(path,
                               '{kind}-labels-idx1-ubyte.gz'.format(kind=kind))
    images_path = os.path.join(path,
                               '{kind}-images-idx3-ubyte.gz'.format(kind=kind))

    with gzip.open(labels_path, 'rb') as label_file:
        struct.unpack('>II', label_file.read(8))
        labels = np.frombuffer(label_file.read(), dtype=np.uint8)

    n_images = len(labels)
    with gzip.open(images_path, 'rb') as image_file:
        struct.unpack('>IIII', image_file.read(16))
        images = np.frombuffer(image_file.read(), dtype=np.uint8).reshape(n_images, 28, 28)

    return images, labels


def fetch_fashion_images(kind='train'):
    data_home = get_data_home()

    fashion_dir = os.path.join(data_home, DATA_NAME)

    if not os.path.exists(fashion_dir):
        download_fashion_mnist(fashion_dir)

    images, labels = load_fashion(fashion_dir, kind=kind)

    return images, labels
