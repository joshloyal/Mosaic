from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

import pandas as pd
from PIL import Image as pil_image

from image_vis import contexts
from image_vis import image_io
from image_vis import features
from image_vis import plots


__all__ = ['image_histogram']


def image_histogram(x, y,
                    data,
                    n_bins=30,
                    image_col=None,
                    image_dir='',
                    n_samples=None,
                    image_size=(20, 20),
                    random_state=123,
                    n_jobs=1,
                    **kwargs):
    """Create an image histogram binned by the `x`.

    Parameters
    ----------
    image_col : str
        Name of the column pointing to the image files

    x : str
        Name of the column bin the x-axis.

    y : str
        Name of the column to sort they values. No sorting is performed
        if y is None.

    data : pandas.DataFrame
        The dataframe where both columns are present.

    thumbnail_size : int
        The size of each image in the histogram.

    image_dir: str
        Path to the directory holding the images.

    n_samples : int (default=None)
        The number of samples do downsample the dataset to.

    random_state : int
        The seed to use for the random number generator.

    n_jobs : int (default=1)
        The number of parallel workers to use for loading
        the image files.

    Examples
    --------

    Create an image histogram.

    .. plot:: ../examples/image_histogram.py
    """
    data = data.copy()
    if n_samples is not None and n_samples < len(data):
        data = data.sample(n_samples, replace=True, random_state=random_state)

    if not image_dir:
        image_dir = contexts.get_image_dir()

    if not image_col:
        image_col = contexts.get_image_col()

    if y in features.HSVFeatures.all_features():
        images = image_io.load_images(
            data[image_col],
            image_dir=image_dir,
            image_size=image_size,
            as_image=True,
            n_jobs=n_jobs)
        hsv = features.extract_hsv_stats(images, n_jobs=n_jobs)
        data[y] = hsv[:, features.HSVFeatures.feature_index(y)]

    data['x_bin'] = pd.cut(data[x], n_bins, labels=False)
    bin_max = data.groupby('x_bin').size().max()

    px_w = image_size[0] * n_bins
    px_h = image_size[1] * bin_max

    #background_color = (50, 50, 50)
    background_color = (255, 255, 255)
    canvas = pil_image.new('RGB', (px_w, px_h), background_color)

    thumbnail_px = image_size
    bins = list(set(list(data.x_bin)))

    for item in bins:
        tmp = data[data.x_bin == item].copy()

        # sort y values if present
        if y is not None:
            tmp.sort_values(by=y, ascending=False, inplace=True)

        tmp.reset_index(drop=True, inplace=True)

        y_coord = px_h
        x_coord = image_size[0] * item

        for i in range(len(tmp.index)):
            thumbnail = image_io.load_image(
                tmp[image_col].iloc[i],
                image_dir,
                image_size=image_size,
                as_image=True)
            canvas.paste(thumbnail, (x_coord, y_coord))
            y_coord -= image_size[1]

    return plots.pillow_to_matplotlib(canvas, **kwargs)
