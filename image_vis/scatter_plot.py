from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

import numpy as np
import pandas as pd
from PIL import Image as pil_image

from image_vis import image_io


__all__ = ['scatter_plot']


def scatter_plot(image_col,
                 x, y,
                 data,
                 thumbnail_size=5,
                 image_dir='',
                 n_samples=None,
                 fig_size=(500, 500),
                 random_state=123):
    """Create an image scatter plot based on columns `y` vs `x`.

    Parameters
    ----------
    image_col : str
        Name of the column pointing to the image files

    x : str
        Name of the column to use for the x-axis.

    y : str
        Name of the column to use for the y-axis

    data : pandas.DataFrame
        The dataframe where both columns are present.

    thumbnail_size : int
        The size of each image in the scatter plot.

    image_dir : str
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

    x_var = data[x].values
    y_var = data[y].values

    # scale the variables between 0-1
    x_var /= np.abs(x_var).max()
    y_var /= np.abs(y_var).max()

    # now stretch them to fit the canvas
    fig_width, fig_height = fig_size
    padding = 0
    x_var = np.floor(x_var * (fig_width / 2 - padding) + fig_width / 2)
    y_var = np.floor(y_var * (fig_height / 2 - padding) + fig_height / 2)

    background_color = (255, 255, 255)
    canvas = pil_image.new('RGB', fig_size, background_color)

    for i in range(len(data[image_col])):
        image_loc = image_io.image_path(data[image_col].iloc[i], image_dir)
        point_img = pil_image.open(image_loc).convert('RGB')
        point_img = point_img.resize(
            (thumbnail_size, thumbnail_size), pil_image.LANCZOS)
        point_width, point_height = point_img.size


        width_pad = 0 if not point_width % 2 else 1
        height_pad = 0 if not point_height % 2 else 1
        bounding_box = (
            int(x_var[i] - point_width // 2),
            int(y_var[i] - point_height // 2),
            int(x_var[i] + point_width // 2 + width_pad),
            int(y_var[i] + point_height // 2 + height_pad)
        )
        canvas.paste(point_img, bounding_box)

    return canvas
