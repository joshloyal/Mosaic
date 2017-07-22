import glob
import os
import itertools
import urllib.request as url_request
import tarfile
import json

import pickle
import numpy as np
import pandas as pd
from sklearn.utils import check_random_state

from image_vis import image_io
from image_vis.datasets.base import get_data_home, ImageDataBundle
from image_vis.datasets.progress_bar import chunk_read


HERE = os.path.dirname(os.path.abspath(__file__))
URL = "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
ARCHIVE_NAME = "cifar-10-batches-py.tar.gz"
DATA_NAME = "cifar-10-batches-py"

def download_cifar10_images(target_dir):
    archive_path = os.path.join(target_dir, ARCHIVE_NAME)

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    print(80 * '-')
    print("Downloading cifar10 images to {}".format(archive_path))
    opener = url_request.urlopen(URL)
    with open(archive_path, 'wb') as f:
        f.write(chunk_read(opener))

    print(80 * '-')
    print("Extracting cifar10 images to {}".format(target_dir))
    tarfile.open(archive_path, "r:gz").extractall(path=target_dir)

    os.remove(archive_path)


def fetch_cifar10_images():
    data_home = get_data_home()

    cifar10_dir = os.path.join(data_home, DATA_NAME)

    if not os.path.exists(cifar10_dir):
        download_cifar10_images(data_home)

    metadata_path = os.path.join(cifar10_dir, 'batches.meta')
    with open(metadata_path, 'rb') as metadata_pkl:
        metadata = pickle.load(metadata_pkl, encoding='bytes')
        label_list = metadata[b'label_names']

    n_train_samples = 50000
    images = np.zeros((n_train_samples, 3, 32, 32), dtype='uint8')
    labels = []
    for i in range(1, 6):
        batch_path = os.path.join(cifar10_dir, 'data_batch_' + str(i))

        with open(batch_path, 'rb') as batch_pkl:
            batch_data = pickle.load(batch_pkl, encoding='bytes')
            batch_images = batch_data[b'data'].reshape(-1, 3, 32, 32)
            batch_labels = [str(label_list[label], 'utf-8') for
                            label in batch_data[b'labels']]

            images[(i - 1) * 10000: i * 10000, :, :, :] = batch_images
            labels.extend(batch_labels)

    # we expect images in width X height X channel order
    images = images.transpose(0, 2, 3, 1)
    labels = np.asarray(labels)

    return images, labels
