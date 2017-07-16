import numpy as np
from joblib import Parallel, delayed
from skimage import color

from image_vis import image_io


__all__ = ['extract_hsv_stats']


def hsv_features_single(image, agg_func=np.mean, background=None):
    """For each hsv value (hue, saturation, value) calculate
    an aggregate statistic (`agg_func`) of that value for a image.

    Parameters
    ----------
    image : np.array of shape (width, height, 3)
        The image over which the aggregte hsv statistics
        are calculated.
    agg_func : numpy ufunc (default=np.mean)
        A function that will calculate a scalar statistic
        over the given values.
    background : array-like of shape [3,] (default=None)
        The background color value for each hsv channel.
        These values will be masked out in the calculation.
        If None, then all values are included in the statistics
        calculation.

    Returns
    -------
    tuple (h_mean, s_mean, v_mean):
        The statitics for each channel.
    """
    image = np.asarray(image, dtype=np.uint8)
    hsv_image = color.rgb2hsv(image)

    if background is not None:
        h_channel = hsv_image[:, :, 0]
        h_mean = agg_func(
            np.ma.array(h_channel, mask=(h_channel == background[0]))
        )
        h_mean = background[0] if h_mean is np.ma.masked else h_mean

        s_channel = hsv_image[:, :, 1]
        s_mean = agg_func(
            np.ma.array(s_channel, mask=(s_channel == background[1]))
        )
        s_mean = background[1] if s_mean is np.ma.masked else s_mean

        v_channel = hsv_image[:, :, 2]
        v_mean = agg_func(
            np.ma.array(v_channel, mask=(v_channel == background[2]))
        )
        v_mean = background[2] if v_mean is np.ma.masked else v_mean
    else:
        h_mean = agg_func(hsv_image[:, :, 0])
        s_mean = agg_func(hsv_image[:, :, 1])
        v_mean = agg_func(hsv_image[:, :, 2])

    return h_mean, s_mean, v_mean


def extract_hsv_stats(image_list, mode='mean', background=None, n_jobs=1):
    """Extract aggregate statistics in the HSV domain of an RGB image.

    A useful ordering tool is the HSV values of an RGB image.
    In particular, arranging images by H (hue) will order them by
    their color along the color spectrum. This function extracts
    a scalar statistic for each HSV channel of every image in an array.

    Parameters
    ----------
    image_list : list of lenth [n_samples,]
        A list of PIL.Images.
    mode : str {'mean', 'median'} (default='mean')
        The statistic to extract for each channel.
    background : array-like of shape [3,] or str {'white', 'black']
                 (default=None)
        The background color value for each hsv channel.
        These values will be masked out in the calculation.
        If None, then all values are included in the statistics
        calculation.
    n_jobs : int (default=1)
        Number of jobs to run in parallel.

    Returns
    -------
    np.array of shape [n_samples, 3]
        An array containing the hsv statistics for each channel.
    """
    if background == 'white':
        background = np.array([0, 0, 1], dtype=np.uint8)
    elif background == 'black':
        background = np.array([0, 0, 0], dtype=np.uint8)


    if mode == 'mean':
        agg_func = np.mean
    elif mode == 'median':
        agg_func = np.median
    else:
        raise ValueError("Unkown mode `{}`.".format(mode))

    result = Parallel(n_jobs=n_jobs)(
        delayed(hsv_features_single)(image, agg_func, background)
        for image in image_list)

    return np.vstack(result)
