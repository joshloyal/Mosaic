from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

import numpy as np

from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib import pyplot as plt
import seaborn as sns

from image_vis import features
from image_vis import contexts
from image_vis import image_io
from image_vis import plots


__all__ = ['scatter_plot']


def images_to_scatter(images, x_var, y_var, threshold=None, alpha=0.9,
                      **kwargs):
    """Creates a scatter plot.

    Parameters
    ----------
    images : np.array of shape [n_samples, n_width, n_height, n_channels]
        A 4D array holding the images to plot.

    x_var : np.array of shape [n_samples,]
        The variable to plot on the x-axis

    y_var : np.array of shape [n_samples,]
        The variable to plot on the y-axis

    threshold : float
        In order to avoid clutter only one point in a ball of
        radius `threshold` is displayed. Note that features
        are re-scaled to lie on the unit square [0, 1] x [0, 1].

    alpha : float
        The alpha level for each image.

    Returns
    -------
    ...
    """
    # scale the variables between 0-1
    features.minmax_scale(x_var)
    features.minmax_scale(y_var)
    xy = np.c_[x_var, y_var]

    fig, ax = plt.subplots(**kwargs)

    # something big. points lie in [0, 1] x [0, 1].
    shown_points = np.array([[1., 1.]])

    for i in range(len(images)):
        dist = np.sum((xy[i] - shown_points) ** 2, axis=1)
        if threshold and np.min(dist) < threshold:
            continue
        shown_points = np.r_[shown_points, [xy[i]]]

        ab = AnnotationBbox(OffsetImage(images[i], alpha=alpha),
                            xy[i, :],
                            frameon=False, xycoords='data')
        ax.add_artist(ab)

    return plots.remove_axis(ax=ax)


def scatter_plot(x, y,
                 data,
                 image_dir='',
                 image_col=None,
                 n_samples=None,
                 image_size=(20, 20),
                 hue=None,
                 threshold=None,
                 alpha=0.9,
                 random_state=123,
                 n_jobs=1,
                 **kwargs):
    """Create an image scatter plot based on columns `x` vs. `y`.

    Parameters
    ----------
    x : str
        Name of the column to use for the x-axis.

    y : str
        Name of the column to use for the y-axis

    data : pandas.DataFrame
        The dataframe where both columns are present.

    image_dir : str
        Path to the directory holding the images.

    image_col : str
        Name of the column containing the image files.

    n_samples : int (default=None)
        If not None, then randomly downsample the dataset
        to `n_sample` images.

    image_size : int
        The size of each image in the scatter plot.

    threshold : float (default=None)
        In order to avoid clutter only one point in a ball of
        radius `threshold` is displayed. Note that features
        are re-scaled to lie on the unit square [0, 1] x [0, 1].
        The default of None means all points are displayed.

    random_state : int
        The seed to use for the random number generator.

    n_jobs : int
        The number of parallel jobs used to load the
        images from disk.

    Examples
    --------

    Create a scatter plot with hue labels.

    .. plot:: ../examples/scatter_plot.py
    """
    data = data.copy()
    if n_samples is not None and n_samples < len(data):
        data = data.sample(n_samples, replace=True, random_state=random_state)

    if not image_dir:
        image_dir = contexts.get_image_dir()

    if not image_col:
        image_col = contexts.get_image_col()

    # get co-variates
    x_var = data[x].values
    y_var = data[y].values

    # load images
    images = image_io.load_images(
        data[image_col],
        image_dir=image_dir,
        as_image=False,
        image_size=image_size,
        n_jobs=n_jobs)

    # TODO (seaborn is only required for a color palette. Remove this)
    if hue is not None:
        values, value_map = np.unique(data[hue], return_inverse=True)
        palette = sns.husl_palette(len(values))
        images = [features.color_image(img, hue=palette[val]) for
                  img, val in zip(images, value_map)]

    return images_to_scatter(images, x_var, y_var, threshold=threshold,
                             alpha=alpha, **kwargs)
