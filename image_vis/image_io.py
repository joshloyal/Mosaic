from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

import glob
import os
import itertools

import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from PIL import Image as pil_image

from image_vis import contexts
from image_vis import features as feature_lib


image_extensions = {'jpg', 'jpeg', 'png'}


def image_path(image_file, image_dir=''):
    """Construct the full image path to a file.

    Parameters
    ----------
    image_file : str
        Location of the image file on disk.
    image_dir : str (default='')
        The directory where the image resides on disk. This string will be
        appended to the beginning of the `image_file`.

    Returns
    -------
    str :
        Absolute path of the image on disk.
    """
    return os.path.join(image_dir, image_file)


def image_glob_pattern(image_directory, ext):
    """Glob pattern string used to match all image files with a particular
    extension in a directory.

    Parameters
    ----------
    image_directory : str
        The directory where the images are located.
    ext : str
        The extension of the image files.

    Returns
    -------
    str :
        Glob pattern
    """
    return os.path.join(image_directory, '*.' + ext)


def image_glob(image_directory, ext):
    """Get a list of all image files with a particular extension in a
    directory.

    Parameters
    ----------
    image_directory : str
        The directory where the images are located.
    ext : str
        The extension of the image files.

    Returns
    -------
    list of str:
        List of images matching the constructed glob pattern.
    """
    return glob.glob(image_glob_pattern(image_directory, ext))


def sample_images(images, n_samples, seed=123):
    """Take a random sample without replacement of images from a list of
    images.

    Parameters
    ----------
    images : list of str
        List of images to sample.
    n_samples : int
        Number of samples to take.
    seed : int
        Seed for the random number generator.

    Returns
    -------
    np.array of str:
        An array of sampled images.
    """
    random_state = np.random.RandomState(seed)
    return random_state.choice(images, size=n_samples, replace=False)


def load_image(image_file,
               image_dir='',
               target_size=None,
               as_image=False,
               dtype=np.uint8):
    """Loads an image from a file on disk.

    Support formats are `jpg`, `png`, or `gif`.

    Parameters
    ----------
    image_file : str
        The image file on disk.
    image_dir : str (default='')
        The directory where the image resides on disk. This string will
        be appended to the beginning of `image_file`.
    target_size : tuple (default=None)
        The target size in pixels. This is a 2-tuple (width, height).
        If None then no resizing is performed.
    as_image : bool (default=False)
        Whether to return a PIL Image. If True a PIL Image is returned
        otherwise the output is a numpy array.
    dtype : numpy dtype (default=np.uint8)
        The dtype of the output numpy array.
    """
    image_loc = image_path(image_file, image_dir=image_dir)
    img = pil_image.open(image_loc).convert('RGB')

    if target_size:
        img = img.resize((target_size[1], target_size[0]), pil_image.LANCZOS)

    if as_image:
        return img

    return np.asarray(img, dtype)


def load_images(image_files,
                image_dir='',
                n_samples=None,
                target_size=None,
                as_image=False,
                random_state=123,
                n_jobs=1,
                dtype=np.uint8):
    """Loads images from a file on disk.

    Support formats are `jpg`, `png`, or `gif`.

    Parameters
    ----------
    image_files : list of str
        A list of str pointing to image files on disk.
    image_dir : str (default='')
        The directory where the image resides on disk. This string will
        be appended to the beginning of `image_file`.
    n_samples : int
        Number of samples to take.
    target_size : tuple (default=None)
        The target size in pixels. This is a 2-tuple (width, height).
        If None then no resizing is performed.
    as_image : bool (default=False)
        Whether to return a PIL Image. If True a PIL Image is returned
        otherwise the output is a numpy array.
    random_state : int
        The random state used to seed the random number generator. Only
        used if sampling is performed.
    n_jobs : int (default=1)
        The number parallel jobs to use for loading images. If -1 is
        specified then all cores are utilized.
    dtype : numpy dtype (default=np.uint8)
        The dtype of the output numpy array.
    """
    if n_samples is not None and n_samples < len(image_files):
        image_files = sample_images(image_files, n_samples, seed=random_state)

    # perform this in parallel with joblib
    images = Parallel(n_jobs=n_jobs)(
                delayed(load_image)(img,
                                    image_dir=image_dir,
                                    target_size=target_size,
                                    as_image=as_image,
                                    dtype=dtype)
                for img in image_files)

    if as_image:
        return images

    return np.stack(images, axis=0)


def load_from_directory(image_directory,
                        n_samples=None,
                        target_size=None,
                        as_image=False,
                        random_state=123,
                        n_jobs=1,
                        dtype=np.uint8):
    """Loads images from a directory on disk.

    Support image formats are `jpg`, `png`, or `gif`.

    Parameters
    ----------
    image_directory : str
        The absolute path where images are located on disk.
    n_samples : int
        Number of samples to take.
    target_size : tuple (default=None)
        The target size in pixels. This is a 2-tuple (width, height).
        If None then no resizing is performed.
    as_image : bool (default=False)
        Whether to return a PIL Image. If True a PIL Image is returned
        otherwise the output is a numpy array.
    random_state : int
        The random state used to seed the random number generator. Only
        used if sampling is performed.
    n_jobs : int (default=1)
        The number parallel jobs to use for loading images. If -1 is
        specified then all cores are utilized.
    dtype : numpy dtype (default=np.uint8)
        The dtype of the output numpy array.
    """
    image_files = list(itertools.chain.from_iterable(
        [image_glob(image_directory, ext) for ext in image_extensions]))
    return load_images(image_files,
                       n_samples=n_samples,
                       target_size=target_size,
                       as_image=as_image,
                       random_state=random_state,
                       n_jobs=n_jobs,
                       dtype=dtype)


def directory_to_dataframe(image_dir='',
                           features=None,
                           n_jobs=-1):
    """Create a pandas.DataFrame containing the path to all images in
    a directory.

    The pandas.DataFrame has a single column `image_path`, which contains
    the paths to the various images. The paths are relative to the given
    `image_dir`. For example, a directory of the form
        - image_dir
            |-- image1.jpg
            |-- image2.jpg
    would result in the following  single column dataframe:
        pandas.DataFrame({'image_path': ['image1.jpg', 'image2.jpg']}).

    Parameters
    ----------
    image_dir : str
        The directory to search of images and place their paths
        in a dataframe.
    features : list or None
        A list of features to include in the dataframe. The default (None)
        includes no additional features.

    Returns
    -------
    pandas.DataFrame
        A dataframe containing the image paths.
    """
    if not image_dir:
        image_dir = contexts.get_image_dir()

    image_files = list(itertools.chain.from_iterable(
        [image_glob(image_dir, ext) for ext in image_extensions]))
    image_files = [f.split(image_dir + os.path.sep)[1] for
                   f in image_files]
    data = pd.DataFrame({'image_path': image_files})

    if features:
        if set(features) & set(feature_lib.HSVFeatures.all_features()):
            images = load_images(
                data['image_path'],
                image_dir=image_dir,
                as_image=True,
                n_jobs=n_jobs)
            hsv = feature_lib.extract_hsv_stats(images, n_jobs=n_jobs)
            for feature in features:
                feature_idx = feature_lib.HSVFeatures.feature_index(feature)
                data[feature] = hsv[:, feature_idx]
        else:
            raise ValueError('Unknown features.')

    return data

