from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

import numpy as np
from PIL import Image as pil_image

from image_vis import image_io


def images_to_sprite(images):
    """Creates a sprite image along with any necessary padding.

    Parameters
    ----------
    images : list
        A List of PIL Image objects.

    Returns
    -------
    A properly shaped NxWx3 PIL Image with any necessary padding.
    """
    n_samples = len(images)

    #features = hsv_features(images, background='white', n_jobs=-1)
    #image_order = np.argsort(features[:, 0])

    if n_samples < 1:
        raise ValueError('Cannot create a sprite image from zero images.')

    image_width, image_height = images[0].size

    # sprite image should be sqrt(n_samples) x sqrt(n_samples). If
    # n_samples is not a perfect square then we pad with white images.
    table_size = int(np.ceil(np.sqrt(n_samples)))

    # create the new image. Hard-code the background color to white
    background_color = (255, 255, 255)
    sprite_size = (table_size * image_width, table_size * image_height)
    sprite_image = pil_image.new('RGB', sprite_size, background_color)

    # loop through the images and add them to the sprite image
    for index, image in enumerate(images):
        row_index = int(index / table_size)
        column_index = index % table_size

        # determine the bounding box of the image (where it is)
        left = column_index * image_width
        right = left + image_width
        upper = row_index * image_height
        lower = upper + image_height
        bounding_box = (left, upper, right, lower)

        sprite_image.paste(image, bounding_box)

    return sprite_image


def directory_to_sprites(image_directory,
                         n_samples=None,
                         random_state=123,
                         n_jobs=1):
    """Creates a sprite image along with any necessary padding.

    Parameters
    ----------
    image_directory : str
        Path to the directory holding the images.

    n_samples : int (default=None)
        The number of random sample images to use. If None, then
        all images are loaded. This can be memory expensive.

    as_image : bool (default=False)
        Whether to return a PIL image otherwise return a numpy array.

    random_state : int (default=123)
        The seed to use for the random sampling.

    n_jobs : int (default=1)
        The number of parallel workers to use for loading
        the image files.

    Returns
    -------
    A properly shaped NxWx3 image with any necessary padding.
    """
    images = image_io.load_from_directory(
        image_directory,
        n_samples=n_samples,
        dtype=np.float32,
        as_image=True,
        random_state=random_state,
        n_jobs=n_jobs)

    return images_to_sprite(images)

def list_to_sprites(image_files,
                    image_dir='',
                    n_samples=None,
                    as_image=False,
                    n_jobs=1):
    """Creates a sprite image along with any necessary padding.

    Parameters
    ----------
    image_files : list of str
        List of paths to images.

    image_dir : str
        The common directory where all the images are located.

    n_samples : int (default=None)
        The number of random sample images to use. If None, then
        all images are loaded. This can be memory expensive.

    as_image : bool (default=False)
        Whether to return a PIL image otherwise return a numpy array.

    n_jobs : int (default=1)
        The number of parallel workers to use for loading
        the image files.

    Returns
    -------
    A properly shaped NxWx3 image with any necessary padding.
    """
    images = image_io.load_images(image_files,
                                  image_dir=image_dir,
                                  n_samples=n_samples,
                                  dtype=np.float32,
                                  n_jobs=n_jobs)

    return images_to_sprite(images, as_image=as_image)


def column_to_sprites(image_column,
                      sort_by=None,
                      data=None,
                      image_directory='',
                      n_samples=None,
                      random_state=123,
                      n_jobs=1):
    """Creates a sprite image along with any necessary padding.

    Parameters
    ----------
    image_column : str
        Column name corresponding to the images.

    sort_by : str
        Column to sort by.

    data : pd.DataFrame
        Pandas dataframe holding the dataset.

    image_directory : str (default='')
        The location of the image files on disk.

    n_samples : int (default=None)
        The number of random sample images to use. If None, then
        all images are loaded. This can be memory expensive.

    as_image : bool (default=False)
        Whether to return a PIL image otherwise return a numpy array.

    random_state : int (default=123)
        The seed to use for the random sampling.

    n_jobs : int (default=1)
        The number of parallel workers to use for loading
        the image files.

    Returns
    -------
    A properly shaped NxWx3 image with any necessary padding.
    """
    if n_samples is not None and n_samples < len(data):
        data = data.sample(n=n_samples,
                           replace=False,
                           random_state=random_state)

    if sort_by is not None:
        data = data.sort_values(by=sort_by, ascending=True)

    images = image_io.load_images(
        data[image_column],
        image_dir=image_directory,
        as_image=True,
        n_jobs=n_jobs)

    return images_to_sprite(images)
