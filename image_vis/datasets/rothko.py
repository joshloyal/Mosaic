import glob
import os
import itertools
import urllib.request as url_request
import tarfile
import json

import numpy as np
from sklearn.utils import check_random_state

from image_vis import image_io
from image_vis.datasets.base import get_data_home, ImageDataBundle
from image_vis.datasets.progress_bar import chunk_read


HERE = os.path.dirname(os.path.abspath(__file__))
URL = "https://s3.amazonaws.com/image-vis/rothko_images.tar.gz"
ARCHIVE_NAME = "rothko_images.tar.gz"
DATA_NAME = "rothko_images"

def download_rothko_images(target_dir):
    archive_path = os.path.join(target_dir, ARCHIVE_NAME)

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    print(80 * '-')
    print("Downloading rothko images to {}".format(archive_path))
    opener = url_request.urlopen(URL)
    with open(archive_path, 'wb') as f:
        f.write(chunk_read(opener))

    print(80 * '-')
    print("Extracting rothko images to {}".format(target_dir))
    tarfile.open(archive_path, "r:gz").extractall(path=target_dir)

    os.remove(archive_path)


def fetch_rothko_images():
    data_home = get_data_home()

    rothko_dir = os.path.join(data_home, DATA_NAME)

    if not os.path.exists(rothko_dir):
        download_rothko_images(data_home)

    return rothko_dir, image_io.directory_to_dataframe(rothko_dir)
