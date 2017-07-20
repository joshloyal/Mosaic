import numpy as np
import pandas as pd
import scipy

from image_vis import image_io
from image_vis import contexts


def get_variable(data, var):
    if data is not None:
        var = data.get(var, var)

    if isinstance(var, pd.Series):
        var = var.values

    return var


def get_images(data, images,
               image_dir='',
               as_image=False,
               target_size=None,
               n_jobs=1):
    if isinstance(images, np.ndarray):
        if as_image:
            return [image_io.to_pillow_image(img, target_size=target_size) for
                    img in images]
        elif target_size:
            return [scipy.misc.imresize(img, target_size, interp='lanczos') for
                    img in images]

    if not image_dir:
        image_dir = contexts.get_image_dir()

    if not images:
        images = contexts.get_image_col()

    images = image_io.load_images(
        data[images],
        image_dir=image_dir,
        as_image=as_image,
        target_size=target_size,
        n_jobs=n_jobs)

    return images
