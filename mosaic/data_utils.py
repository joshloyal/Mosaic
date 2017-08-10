import numpy as np
import pandas as pd
import scipy

from mosaic import image_io
from mosaic import contexts


def get_variable(data, var):
    """Helper function to obtain the variable data either from
    `data` or `var`.

    Parameters
    ----------
    data : pandas.DataFrame
        Tidy ("long-form") dataframe where each column is a variable
        and each row is an observation.

    var : str or array-like
        Data or name of variables in `data`.

    Returns
    -------
    array : np.array
        A numpy array holding the values of `var`.
    """
    if isinstance(var, pd.Series):
        var = var.values
    elif isinstance(var, list):
        var = np.asarray(var)
    elif isinstance(var, np.ndarray):
        var = var
    elif data is not None and var in data:
        var = data[var]
    else:
        raise ValueError('Could not find {}.'.format(var))

    return var


def get_images(data, images,
               image_dir='',
               image_size=None,
               as_image=False,
               index=None,
               n_jobs=1):
    """Helper function to load images from disk or properly format
    an already existing image array.

    Parameters
    ----------
    data : pandas.DataFrame
        Tidy ("long-form") dataframe where each column is a variable
        and each row is an observation.

    images : str or array-like of shape [n_samples, width, height, channels], optional
        Image array or name of the variable containing the image file
        paths within `data`.

    image_dir : str, optional
        The location of the image files on disk. Images will
        be loaded from files matching the pattern
        'image_dir + os.path.sep + image_path'.

    image_size : int, optional
            The size of each image displayed in the scatter plot. Images
            will be sampled to `image_size` if the size of the images
            do not match `image_size`.

    as_image : bool
        If True, the returned images are converted to PIL.Image
        objects.

    Returns
    -------
    images : array-like
        Either a list of PIL.Images or a np.array of shape
        [n_samples, width, height, channels].
    """
    if isinstance(images, np.ndarray):
        if as_image:
            return [image_io.to_pillow_image(img, image_size=image_size) for
                    img in images]
        elif image_size:
            return np.asarray([scipy.misc.imresize(img, image_size, interp='lanczos') for
                    img in images])

    if not image_dir:
        image_dir = contexts.get_image_dir()

    if not images:
        images = contexts.get_image_data()
        if images is None:
            images = data[contexts.get_image_col()]

    if index is not None:
        images = images.iloc[index]

    images = image_io.load_images(
        images,
        image_dir=image_dir,
        as_image=as_image,
        image_size=image_size,
        n_jobs=n_jobs)

    return images
