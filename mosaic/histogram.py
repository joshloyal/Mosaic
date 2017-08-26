from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image as pil_image

from mosaic import data_utils
from mosaic import contexts
from mosaic import image_io
from mosaic import features
from mosaic import plots


__all__ = ['image_histogram']


def images_to_histogram(images, x, n_bins=None, sort_by=None):
    """Create an image histogram.

    Parameters
    ----------
    images : listof PIL Images.
        Images to display in the image histogram. All images must be
        the same shape.

    x : np.array of shape [n_samples,]
        The variable whose histogram is displayed.

    n_bins : int or None, optional
        Specification of the number of bins. If None, then the
        Freedman-Diaconis estimator is used to determine the number of bins.

    sort_by : np.array of shape [n_samples,], optional
        Data or name of the variable to sort images by on the y-axis.

    Returns
    -------
    A properly shaped width x height x 3 PIL Image.
    """
    n_bins = n_bins if n_bins is not None else 'fd'
    hist, bin_edges = np.histogram(x, bins=n_bins)
    n_bins = hist.shape[0]
    bin_max = hist.max()

    width, height = images[0].size
    px_w = width * n_bins
    px_h = height * bin_max

    #background_color = (50, 50, 50)
    background_color = (255, 255, 255)
    canvas = pil_image.new('RGB', (px_w, px_h), background_color)

    thumbnail_px = (width, height)

    for bin_idx, edge in enumerate(zip(bin_edges, bin_edges[1:])):
        edge_mask = (x >= edge[0]) & (x < edge[1])
        tmp_sort = sort_by[edge_mask]
        tmp = [images[index] for index in np.where(edge_mask)[0]]

        # sort y values if present
        if sort_by is not None:
            tmp = [tmp[index] for index in np.argsort(tmp_sort)[::-1]]

        y_coord = px_h
        x_coord = width * bin_idx

        for thumbnail in tmp:
            canvas.paste(thumbnail, (x_coord, y_coord))
            y_coord -= height

    return canvas


def histogram_matplotlib(images, x, n_bins=None, sort_by=None, **kwargs):
    fig, ax = plt.subplots(**kwargs)

    n_bins = n_bins if n_bins is not None else 'fd'
    hist, bin_edges = np.histogram(x, bins=n_bins)
    n_bins = hist.shape[0]
    bin_max = hist.max()

    y_max = 0
    for bin_idx, edge in enumerate(zip(bin_edges, bin_edges[1:])):
        img_height = abs(edge[1] - edge[0])
        edge_mask = (x >= edge[0]) & (x < edge[1])
        bin_images = images[edge_mask]

        # sort y values if present
        if sort_by is not None:
            bin_sort = sort_by[edge_mask]
            bin_images = bin_images[np.argsort(bin_sort)]

        left, right = edge
        for i, img in enumerate(bin_images):
            bottom = img_height * i
            top = bottom + img_height
            plots.imshow(img, extent=[left, right, bottom, top], interpolation='lanczos')
            if top > y_max:
                y_max = top

    ax.set_xlim(bin_edges[0], bin_edges[-1])
    ax.set_ylim(0, y_max)
    ax.yaxis.set_visible(False)

    return sns.despine(ax=ax, left=True)


def image_histogram(x,
                    images=None,
                    data=None,
                    n_bins=None,
                    sort_by=features.HSVFeatures.SATURATION,
                    image_dir='',
                    image_size=(20, 20),
                    n_jobs=1,
                    **kwargs):
    """Create an univariate image histogram binned by the `x`
    variable.

    Parameters
    ----------
    x : str or array-like of shape [n_samples,]
        Data or names of variables in `data`.

    images : str or array-like of shape [n_samples, width, height, channels], optional
        Image array or name of the variable containing the image file
        paths within `data`.

    data : pandas.DataFrame, optional
        Tidy ("long-form") dataframe where each column is a variable
        and each row is an observation. If `images`, `x`, or `sort_by`
        is a variable name, then it should be contained in `data`.

    n_bins : int or None
        Specification of the number of bins. If None, then the
        Freedman-Diaconis estimator is used to determine the number of bins.

    sort_by : str, HSVFeatures enum or array-like of shape [n_samples,], optional
        Data or name of the variable to sort images by on the y-axis.

    image_dir : str (default='')
        The location of the image files on disk.

    image_size : int
        The size of each image in the scatter plot.

    n_jobs : int (default=1)
        The number of parallel workers to use for loading
        the image files.

    Returns
    -------
    ax : matplotlib Axes
        Returns the Axes object with the plot for further tweaking.


    Examples
    --------

    Create an image histogram.

    .. plot:: ../examples/image_histogram.py
    """
    images = data_utils.get_images(
        data, images,
        image_dir=image_dir,
        image_size=image_size,
        index=None,#x.index,
        as_image=False,
        n_jobs=n_jobs)

    x = data_utils.get_variable(data, x)

    if sort_by is not None:
        if sort_by in features.HSVFeatures.all_features():
            hsv = features.extract_hsv_stats(images, n_jobs=n_jobs)
            sort_by = hsv[:, features.HSVFeatures.feature_index(sort_by)]
        else:
            sort_by = data_utils.get_variable(data, sort_by)

    #histo = images_to_histogram(images, x, n_bins=n_bins, sort_by=sort_by)
    #return plots.pillow_to_matplotlib(histo, **kwargs)

    return histogram_matplotlib(images, x, n_bins=n_bins, sort_by=sort_by, **kwargs)
