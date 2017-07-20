from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

import numpy as np
import sklearn.metrics as metrics
import seaborn as sns

from image_vis import features
from image_vis import contexts
from image_vis import image_io
from image_vis import plots
from image_vis import data_utils
from image_vis.mosaic import images_to_mosaic


__all__ = ['distance_grid']


def images_to_grid(images, x_var, y_var, **kwargs):
    """Creates a grid plot.

    Parameters
    ----------
    images : list of length [n_samples,]
        A List of PIL Image objects. All images must be
        the same shape NxWx3.

    x_var : np.array of shape [n_samples,]
        The x-coordinate in euclidean space.

    y_var : np.array of shape [n_samples,]
        The y-coordinate in euclidean space.

    Returns
    -------
    A properly shaped width x height x 3 PIL Image.
    """
    # scale the variables between 0-1 (subtract off min?)
    features.minmax_scale(x_var)
    features.minmax_scale(y_var)
    xy = np.c_[x_var, y_var]

    # make a grid of evenly spaced points on the grid.
    # The grid is of size sqrt(n_samples) x sqrt(n_samples)
    grid_size = int(np.ceil(np.sqrt(len(images))))
    grid_1d = np.linspace(0, 1, grid_size)
    grid_2d = np.dstack(np.meshgrid(grid_1d, grid_1d)).reshape(-1, 2)

    # distances between the evenly spaced grid and the points
    dist = metrics.euclidean_distances(grid_2d, xy)

    # determine order based on nearest neighbors
    image_order = []
    for i in range(grid_2d.shape[0]):
        index = np.argmin(dist[i, :])
        image_order.append(index)
        dist[:, index] = np.inf  # set to inf so we don't pick this point again
    images = [images[index] for index in image_order]

    grid = images_to_mosaic(images)
    return plots.pillow_to_matplotlib(grid, **kwargs)


def distance_grid(x, y,
                  images=None,
                  data=None,
                  hue=None,
                  image_dir='',
                  image_size=(20, 20),
                  n_jobs=1,
                  **kwargs):
    """Draw a plot ordering images in a regularly spaced 2-d grid
    based on their distance in the x-y plane. The distance between
    points is assumed to be euclidean.

    Parameters
    ----------
    x, y : str or array-like
        Data or names of variables in `data`.
        These variables correspond to the x-y coordinates
        in the euclidean space.

    images : str or array-like
        Image arrays or names of the column pointing to the
        image paths within `data`.

    data : pd.DataFrame
        Pandas dataframe holding the dataset.

    hue : str or array-like
        Data or the name of the variable to use to color
        the individual images on the grid.

    image_dir : str (default='')
        The location of the image files on disk.

    image_size : int
        The size of each image in the scatter plot.

    n_jobs : int (default=1)
        The number of parallel workers to use for loading
        the image files.

    Returns
    -------
    A properly shaped NxWx3 image with any necessary padding.

    Examples
    --------

    Create a grid plot with hue labels.

    .. plot:: ../examples/distance_grid.py
    """
    x_var = data_utils.get_variable(data, x)
    y_var = data_utils.get_variable(data, y)

    # TODO (seaborn is only required for a color palette. Remove this)
    if hue is not None:
        images = data_utils.get_images(
            data, images,
            image_dir=image_dir,
            as_image=False,
            image_size=image_size,
            n_jobs=n_jobs)

        hue = data_utils.get_variable(data, hue)
        values, value_map = np.unique(hue, return_inverse=True)
        palette = sns.husl_palette(len(values))
        images = [features.color_image(img, hue=palette[val]) for
                  img, val in zip(images, value_map)]
        images = [image_io.to_pillow_image(img) for img in images]
    else:
        # load images
        images = data_utils.get_images(
            data, images,
            image_dir=image_dir,
            as_image=True,
            image_size=image_size,
            n_jobs=n_jobs)

    return images_to_grid(images, x_var, y_var, **kwargs)
