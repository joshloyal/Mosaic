from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

import numpy as np

from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib import pyplot as plt
import seaborn as sns

from mosaic import data_utils
from mosaic import contexts
from mosaic import features
from mosaic import image_io
from mosaic import plots


__all__ = ['scatter_plot']


def images_to_scatter(images, x, y, threshold=None, alpha=0.9,
                      **kwargs):
    """Creates a scatter plot.

    Parameters
    ----------
    images : np.array of shape [n_samples, n_width, n_height, n_channels]
        A 4D array holding the images to plot.

    x : np.array of shape [n_samples,]
        The variable to plot on the x-axis

    y : np.array of shape [n_samples,]
        The variable to plot on the y-axis

    threshold : float
        In order to avoid clutter only one point in a ball of
        radius `threshold` is displayed. Note that features
        are re-scaled to lie on the unit square [0, 1] x [0, 1].

    alpha : float
        The alpha level for each image.

    Returns
    -------
    ax : matplotlib Axes
        Returns the Axes object with the plot for further tweaking.
    """
    # scale the variables between 0-1
    xy = np.c_[x, y]

    fig, ax = plt.subplots(**kwargs)

    # something big. points lie in [0, 1] x [0, 1].
    shown_points = np.array([[np.inf, np.inf]])

    for i in range(len(images)):
        dist = np.sum((xy[i] - shown_points) ** 2, axis=1)
        if threshold and np.min(dist) < threshold:
            continue
        shown_points = np.r_[shown_points, [xy[i]]]

        ab = AnnotationBbox(OffsetImage(images[i], alpha=alpha),
                            xy[i, :],
                            frameon=False, xycoords='data')
        ax.add_artist(ab)

    plt.xlim(x.min(), x.max())
    plt.ylim(y.min(), y.max())
    return plots.remove_axis(ax=ax)


def scatter_plot(x, y,
                 images=None,
                 data=None,
                 hue=None,
                 image_dir='',
                 image_size=None,
                 threshold=None,
                 alpha=0.9,
                 color=None,
                 n_jobs=1,
                 **kwargs):
    """Create an image scatter plot based on columns `x` vs. `y`.

    Parameters
    ----------
    x, y : str or array-like
        Data or names of variables in `data`. These variables are
        used for the x and y axes respectively.

    images : str or array-like, optional
        Image arrays or names of the column pointing to the
        image paths within `data`.

    data : pandas.DataFrame, optional
        Tidy ("long-form") dataframe where each column is a variable
        and each row is an observation. If `images` is a variable name,
        then it should be contained in `data`.

    image_dir : str, optional
        The location of the image files on disk. Images will
        be loaded from files matching the pattern
        'image_dir + os.path.sep + image_path'.

    image_size : int, optional
        The size of each image displayed in the scatter plot. Images
        will be sampled to `image_size` if the size of the images
        do not match `image_size`.

    threshold : float, optional
        In order to avoid clutter only one point in a ball of
        radius `threshold` is displayed. Note that features
        are re-scaled to lie on the unit square [0, 1] x [0, 1].
        The default of None means all points are displayed.

    alpha : float, optional
        Alpha level used when displaying images.

    n_jobs : int
        The number of parallel jobs used to load the
        images from disk.

    Examples
    --------

    Create a scatter plot with hue labels.

    .. plot:: ../examples/scatter_plot.py
    """
    # get co-variates
    x = data_utils.get_variable(data, x)
    y = data_utils.get_variable(data, y)

    # load images
    images = data_utils.get_images(
        data, images,
        image_dir=image_dir,
        as_image=False,
        image_size=image_size,
        n_jobs=n_jobs)

    # TODO (seaborn is only required for a color palette. Remove this)
    if hue is not None:
        hue = data_utils.get_variable(data, hue)
        values, value_map = np.unique(hue, return_inverse=True)
        palette = sns.husl_palette(len(values))
        images = [features.color_image(img, hue=palette[val]) for
                  img, val in zip(images, value_map)]
    elif color is not None:
        images = [features.color_image(img, hue=color) for
                  img in images]

    return images_to_scatter(images, x, y, threshold=threshold,
                             alpha=alpha, **kwargs)
