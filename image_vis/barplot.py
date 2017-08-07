from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

import numpy as np
import seaborn as sns

from matplotlib import pyplot as plt

from image_vis import data_utils
from image_vis import features
from image_vis import plots


__all__ = ['image_barplot']


def gen_splits(array, n_elements):
    """Split the horizontal bars with n_elements equally."""
    current_index = 0
    next_index = min(n_elements, len(array))
    while True:
        if current_index >= len(array):
            break
        yield array[current_index:next_index]
        current_index = next_index
        next_index = min(current_index + n_elements, len(array))


def images_to_barplot(images, y, bar_height=30, **kwargs):
    """Create a image bar plot.

    Parameters
    ----------
    images : np.array of shape [n_samples, n_width, n_height, n_channels]
        A 4D array holding the images to plot.

    y : np.array of shape [n_samples,]
        The categorical variable to plot on the y-axis

    bar_height : int
        The number of images placed in a single horizontal bar before
        creating a new bar.

    Returns
    -------
    ax : matplotlib Axes
        Returns the Axes object with the plot for further tweaking.
    """
    fig, ax = plt.subplots(**kwargs)

    img_width, img_height = images.shape[1], images.shape[2]
    vertical_padding = img_width
    horizontal_padding = 1  # this padding is in terms of # of images

    total_n_splits = 0
    max_width = 0
    ticks = []
    labels = np.unique(y)
    for label_idx, label in enumerate(labels):
        image_group = np.squeeze(images[y == label])
        bottom = img_height * total_n_splits + vertical_padding * label_idx
        bottom_start = bottom
        for image_split in gen_splits(image_group, bar_height):
            total_n_splits += 1
            bottom += img_height
            top = bottom + img_height
            for img_idx in range(len(image_split)):
                img = image_split[img_idx]

                left = img_width * (img_idx + horizontal_padding)
                right = img_width * (img_idx + horizontal_padding + 1)
                plt.imshow(img, extent=[left, right, bottom, top])

                if right > max_width:
                    max_width = right

        # the ticks should be in the center of the bar
        ticks.append((top + img_height + bottom_start) / 2.)

    # limits with some wiggle room
    plt.xlim(0, max_width * 1.1)
    plt.ylim(0, top)

    # labels for y-axis
    ax.set_yticklabels([str(i) for i in labels])
    sns.despine(ax=ax, top=True, left=True, right=True, bottom=True)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks(ticks)

    return ax


def image_barplot(y,
                  images=None,
                  data=None,
                  sort_by=None,
                  bar_height=50,
                  image_dir='',
                  image_size=(40, 40),
                  n_jobs=1,
                  **kwargs):
    """Create a barplot where the bars are created from images in the dataset.

    This visualization is useful for analyzing the relationship between images
    and categorical variables.

    Parameters
    ----------
    y : str or array-like
        Data or the name of a variables in `data`. This variable is assumed
        to be categoricel, e.g. discrete, and is used to split the bars
        on the vertical axis.

    images : str or array-like of shape [n_samples, width, height, channels], optional
        Image array or name of the variable containing the image file
        paths within `data`.

    data : pandas.DataFrame, optional
        Tidy ("long-form") dataframe where each column is a variable
        and each row is an observation. If `images` is a variable name,
        then it should be contained in `data`.

    sort_by : str or array-like of shape [n_samples,], optional
        Data or name of the variable to sort images by in the horizontal
        direction.

    bar_height : int
        The number of images placed in a single horizontal bar before
        creating a new bar.

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


    Examples
    --------
    Create a image barplot.

    .. plot:: ../examples/image_barplot.py
    """
    y = data_utils.get_variable(data, y)
    images = data_utils.get_images(data, images,
                                   as_image=False,
                                   image_size=image_size,
                                   n_jobs=n_jobs)

    if sort_by is not None:
        if sort_by in features.HSVFeatures.all_features():
            hsv = features.extract_hsv_stats(images, n_jobs=n_jobs)
            sort_by_values = hsv[:, features.HSVFeatures.feature_index(sort_by)]
            images = images[np.argsort(sort_by_values)]
        else:
            sort_by = data_utils.get_variable(data, sort_by)
            images = images[np.argsort(sort_by)]

    return images_to_barplot(images, y, bar_height=bar_height, **kwargs)
