from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

import pandas as pd
from PIL import Image as pil_image

from image_vis import image_io


def histogram(image_column,
              x_column,
              y_column,
              data,
              n_bins=100,
              thumbnail_size=50,
              image_directory='',
              n_samples=None,
              fig_size=(1000, 1000),
              random_state=123):
    """Create an image histogram binned by the `x_column`.

    Parameters
    ----------
    image_column : str
        Name of the column pointing to the image files

    x_column : str
        Name of the column bin the x-axis.

    y_column : str
        Name of the column to sort they values. No sorting is performed
        if y_column is None.

    data : pandas.DataFrame
        The dataframe where both columns are present.

    thumbnail_size : int
        The size of each image in the histogram.

    image_directory : str
        Path to the directory holding the images.

    n_samples : int (default=None)
        The number of samples do downsample the dataset to.

    fig_size : tuple
        The (width_px, height_px) of the final image in pixels.

    random_state : int
        The seed to use for the random number generator.
    """
    data = data.copy()
    if n_samples is not None and n_samples < len(data):
        data = data.sample(n_samples, replace=True, random_state=random_state)

    data['x_bin'] = pd.cut(data[x_column], n_bins, labels=False)
    bin_max = data.groupby('x_bin').size().max()

    px_w = thumbnail_size * n_bins
    px_h = thumbnail_size * bin_max

    background_color = (50, 50, 50)
    canvas = pil_image.new('RGB', (px_w, px_h), background_color)

    thumbnail_px = (thumbnail_size, thumbnail_size)
    bins = list(set(list(data.x_bin)))

    for item in bins:
        tmp = data[data.x_bin == item].copy()

        # sort y values if present
        if y_column is not None:
            tmp.sort_values(by=y_column, ascending=False, inplace=True)

        tmp.reset_index(drop=True, inplace=True)

        y_coord = px_h
        x_coord = thumbnail_size * item

        for i in range(len(tmp.index)):
            image_loc = image_io.image_path(tmp[image_column].iloc[i], image_directory)
            thumbnail = pil_image.open(image_loc)
            thumbnail.thumbnail(thumbnail_px, pil_image.BICUBIC)
            canvas.paste(thumbnail, (x_coord, y_coord))
            y_coord -= thumbnail_size

    if fig_size:
        canvas.thumbnail(fig_size, pil_image.BICUBIC)

    return canvas
