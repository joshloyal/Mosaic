from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

import numpy as np
from PIL import Image as pil_image

from image_vis import contexts
from image_vis import image_io
from image_vis import features
from image_vis import plots


__all__ = ['mosaic_plot']


def images_to_mosaic(images):
    """Creates a mosaic plot along with any necessary padding.

    Parameters
    ----------
    images : list
        A List of PIL Image objects. All images must be
        the same shape NxWx3.

    Returns
    -------
    A properly shaped NxWx3 PIL Image with any necessary padding.
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

        mosaic_image.paste(image, bounding_box)

    return mosaic_image


def mosaic_plot(image_col,
                data,
                sort_by=None,
                image_dir='',
                target_size=(100, 100),
                fig_size=(500, 500),
                n_samples=None,
                random_state=123,
                n_jobs=1,
                **kwargs):
    """Creates a mosaic plot along with any necessary padding.

    Parameters
    ----------
    image_col : str
        Column name corresponding to the images.

    data : pd.DataFrame
        Pandas dataframe holding the dataset.

    sort_by : str
        Column to sort by.

    image_dir : str (default='')
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

    Examples
    --------

    Create a mosaic plot.

    .. plot:: ../examples/mosaic_plot.py

    Create a mosaic plot with custom ordering on MNIST.

    .. plot:: ../examples/mosaic_plot_custom_order.py
    """
    if n_samples is not None and n_samples < len(data):
        data = data.sample(n=n_samples,
                           replace=False,
                           random_state=random_state)

    if (sort_by is not None and
            sort_by not in features.HSVFeatures.all_features()):
        data = data.sort_values(by=sort_by, ascending=True)

    if not image_dir:
        image_dir = contexts.get_image_dir()

    images = image_io.load_images(
        data[image_col],
        image_dir=image_dir,
        as_image=True,
        target_size=target_size,
        n_jobs=n_jobs)

    if sort_by in features.HSVFeatures.all_features():
        hsv = features.extract_hsv_stats(images, n_jobs=n_jobs)
        sort_by_values = hsv[:, features.HSVFeatures.feature_index(sort_by)]
        sorted_indices = np.argsort(sort_by_values)
        images = [images[i] for i in sorted_indices]

    mosaic = images_to_mosaic(images)

    if fig_size:
        mosaic.thumbnail(fig_size, pil_image.BICUBIC)

    return plots.pillow_to_matplotlib(mosaic, **kwargs)
