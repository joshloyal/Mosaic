import glob
import os
import itertools
import urllib.request as url_request
import tarfile
import json

import numpy as np
import pandas as pd
from sklearn.utils import check_random_state

from mosaic import image_io
from mosaic.datasets.base import get_data_home, ImageDataBundle
from mosaic.datasets.progress_bar import chunk_read


HERE = os.path.dirname(os.path.abspath(__file__))
URL = "https://s3.amazonaws.com/image-vis/mnist.tar.gz"
ARCHIVE_NAME = "mnist.tar.gz"
DATA_NAME = "mnist"

def download_mnist_images(target_dir):
    archive_path = os.path.join(target_dir, ARCHIVE_NAME)

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    print(80 * '-')
    print("Downloading mnist images to {}".format(archive_path))
    opener = url_request.urlopen(URL)
    with open(archive_path, 'wb') as f:
        f.write(chunk_read(opener))

    print(80 * '-')
    print("Extracting mnist images to {}".format(target_dir))
    tarfile.open(archive_path, "r:gz").extractall(path=target_dir)

    os.remove(archive_path)


def fetch_mnist_images():
    data_home = get_data_home()

    mnist_dir = os.path.join(data_home, DATA_NAME)

    if not os.path.exists(mnist_dir):
        download_mnist_images(data_home)

    return mnist_dir, pd.read_csv(os.path.join(mnist_dir, 'mnist_data.csv'))
