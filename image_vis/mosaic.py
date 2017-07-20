from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

import numpy as np
from PIL import Image as pil_image

from image_vis import contexts
from image_vis import data_utils
from image_vis import image_io
from image_vis import features
from image_vis import plots


__all__ = ['mosaic_plot']


def images_to_mosaic(images):
    """Create a mosaic plot of images.

    Parameters
    ----------
    images : listof PIL Images.
        Images to display in the mosaic plot. All images must be
        the same shape.

    Returns
    -------
    ax : matplotlib Axes
        Returns the Axes object with the plot for further tweaking.
    """
    n_samples = len(images)

    if n_samples < 1:
        raise ValueError('Cannot create a sprite image from zero images.')

    image_width, image_height = images[0].size

    # mosaic plot should be sqrt(n_samples) x sqrt(n_samples). If
    # n_samples is not a perfect square then we pad with white images.
    table_size = int(np.ceil(np.sqrt(n_samples)))

    # create the new image. Hard-code the background color to white
    background_color = (255, 255, 255)
    mosaic_size = (table_size * image_width, table_size * image_height)
    mosaic_image = pil_image.new('RGB', mosaic_size, background_color)

    # loop through the images and add them to the mosaic plot
    for index, image in enumerate(images):
        row_index = int(index / table_size)
        column_index = index % table_size

        # determine the bounding box of the image (where it is)
        left = column_index * image_width
        right = left + image_width
        upper = row_index * image_height
        lower = upper + image_height
        bounding_box = (left, upper, right, lower)

        try:
            mosaic_image.paste(image, bounding_box)
        except ValueError:
            raise ValueError(
                'Not all images have the same width and height. '
                'You can force even sizes by setting the `image_size`'
                'argument to the desired dimensions.')

    return mosaic_image


def mosaic_plot(images=None,
                data=None,
                sort_by=None,
                image_dir='',
                image_size=(40, 40),
                n_jobs=1,
                **kwargs):
    """Create a mosaic plot of images.

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
    distance_grid : Combines a mosaic plot with a :func:`scatter_plot`.

    Examples
    --------

    Create a mosaic plot.

    .. plot:: ../examples/mosaic_plot.py

    Create a mosaic plot with custom ordering on MNIST.

    .. plot:: ../examples/mosaic_plot_mnist.py
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

    mosaic = images_to_mosaic(images)

    if 'figsize' in kwargs:
        fig_size = kwargs.pop('figsize')
        mosaic.thumbnail(fig_size, pil_image.BICUBIC)

    return plots.pillow_to_matplotlib(mosaic, **kwargs)
