from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

import numpy as np
from PIL import Image as pil_image

from mosaic import data_utils
from mosaic import features
from mosaic import plots


__all__ = ['image_grid']


def images_to_grid(images, padding=None):
    """Create a grid plot of images.

    Parameters
    ----------
    images : listof PIL Images.
        Images to display in the grid plot. All images must be
        the same shape.

    padding : int, optional
        The padding between images in the grid.

    Returns
    -------
    ax : matplotlib Axes
        Returns the Axes object with the plot for further tweaking.
    """
    n_samples = len(images)

    if n_samples < 1:
        raise ValueError('Cannot create a sprite image from zero images.')

    if padding is None:
        padding = 0

    image_width, image_height = images[0].size

    # grid plot should be sqrt(n_samples) x sqrt(n_samples). If
    # n_samples is not a perfect square then we pad with white images.
    table_size = int(np.ceil(np.sqrt(n_samples)))

    # create the new image. Hard-code the background color to white
    background_color = (255, 255, 255)

    grid_size = (table_size * image_width + (table_size - 1) * padding,
                 table_size * image_height + (table_size - 1) * padding)
    grid_image = pil_image.new('RGB', grid_size, background_color)

    # loop through the images and add them to the grid plot
    for index, image in enumerate(images):
        row_index = int(index / table_size)
        column_index = index % table_size

        # determine the bounding box of the image (where it is)
        left = column_index * (image_width + padding)
        right = left + image_width
        upper = row_index * (image_height + padding)
        lower = upper + image_height
        bounding_box = (left, upper, right, lower)

        try:
            grid_image.paste(image, bounding_box)
        except ValueError:
            raise ValueError(
                'Not all images have the same width and height. '
                'You can force even sizes by setting the `image_size`'
                'argument to the desired dimensions.')

    return grid_image


def image_grid(images=None,
               data=None,
               sort_by=None,
               image_dir='',
               image_size=40,
               padding=None,
               n_jobs=1,
               **kwargs):
    """Create a grid of images.

    Parameters
    ----------
    images : str or array-like of shape [n_samples, width, height, channels], optional
        Image array or name of the variable containing the image file
        paths within `data`.

    data : pandas.DataFrame, optional
        Tidy ("long-form") dataframe where each column is a variable
        and each row is an observation. If `images` is a variable name,
        then it should be contained in `data`.

    sort_by : str or array-like of shape [n_samples,], optional
        Data or name of the variable to sort images by.

    image_dir : str, optional
        The location of the image files on disk. Images will
        be loaded from files matching the pattern
        'image_dir + os.path.sep + image_path'.

    image_size : int, optional
        The size of each image displayed in the scatter plot. Images
        will be sampled to `image_size` if the size of the images
        do not match `image_size`.

    padding : int, optional
        The padding between images in the grid.

    n_jobs : int, optional
        The number of parallel workers to use for loading
        the image files when reading from disk. The default
        uses a single core.

    kwargs : key, value pairings
        Additional keyword arguments are passed to the function used to draw
        the plot on the Axes.

    Returns
    -------
    ax : matplotlib Axes
        Returns the Axes object with the plot for further tweaking.

    See Also
    --------
    scatter_grid : Combines an image grid with a :func:`scatter_plot`.

    Examples
    --------

    Create a image grid.

    .. plot:: ../examples/image_grid.py

    Create a image grid with custom ordering on MNIST.

    .. plot:: ../examples/image_grid_mnist.py
    """
    images = data_utils.get_images(data, images,
                                   as_image=True,
                                   image_size=image_size,
                                   n_jobs=n_jobs)

    if sort_by is not None:
        if sort_by in features.HSVFeatures.all_features():
            hsv = features.extract_hsv_stats(images, n_jobs=n_jobs)
            sort_by_values = hsv[:, features.HSVFeatures.feature_index(sort_by)]
            sorted_indices = np.argsort(sort_by_values)
            images = [images[i] for i in sorted_indices]
        else:
            sort_by = data_utils.get_variable(data, sort_by)
            images = [images[i] for i in np.argsort(sort_by)]

    grid = images_to_grid(images, padding=padding)

    return plots.pillow_to_matplotlib(grid, **kwargs)
