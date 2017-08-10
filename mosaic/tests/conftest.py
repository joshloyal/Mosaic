import itertools
import os
import tempfile
import shutil

import numpy as np
import pytest
import PIL.Image as pil_image


rng = np.random.RandomState(123)


@pytest.fixture(scope='session')
def img_w():
    return 20


@pytest.fixture(scope='session')
def img_h():
    return 20


@pytest.fixture(scope='session')
def image_list(img_w, img_h):
    rgb_images = []
    gray_images = []
    for n in range(8):
        bias = rng.rand(img_w, img_h, 1) * 64
        variance = rng.rand(img_w, img_h, 1) * (255 - 64)
        image_array = rng.rand(img_w, img_h, 3) * variance + bias
        image = pil_image.fromarray(image_array.astype('uint8')).convert('RGB')
        rgb_images.append(image)

        image_array = rng.rand(img_w, img_h, 1) * variance + bias
        image = pil_image.fromarray(image_array.astype('uint8').squeeze()).convert('L')
        gray_images.append(image)

    return [rgb_images, gray_images]


@pytest.fixture(scope='session')
def rgb_image_data(tmpdir_factory, image_list):
    temp_dir = tmpdir_factory.mktemp('data')
    image_paths = []
    for i, image in enumerate(image_list[0]):
        image_file = 'image_{}.jpeg'.format(i)
        image_path = str(temp_dir.join(image_file))
        image.save(image_path)
        image_paths.append(image_file)
    return str(temp_dir), image_paths


@pytest.fixture(scope='session')
def gray_image_data(tmpdir_factory, image_list):
    temp_dir = tmpdir_factory.mktemp('data')
    image_paths = []
    for i, image in enumerate(image_list[1]):
        image_file = 'image_{}.jpeg'.format(i)
        image_path = str(temp_dir.join(image_file))
        image.save(image_path)
        image_paths.append(image_file)
    return str(temp_dir), image_paths


@pytest.fixture(scope='session')
def image_data(tmpdir_factory, image_list):
    temp_dir = tmpdir_factory.mktemp('data')
    image_paths = []
    for i, image in enumerate(itertools.chain(*image_list)):
        image_file = 'image_{}.jpeg'.format(i)
        image_path = str(temp_dir.join(image_file))
        image.save(image_path)
        image_paths.append(image_file)
    return str(temp_dir), image_paths
