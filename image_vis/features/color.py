import numpy as np

from skimage import color


__all__ = ['color_image']


def color_image(img, hue=[1, 0, 0], alpha=0.6):
    """Color an image with the given hue at a specified alpha level.

    This coloring is done by converting the image to the HSV colorspace
    and changing the hue and saturation values to the given hue. The
    value level is kept unchanged.

    Parameters
    ----------
    image : np.array of shape (width, height, 3)
        The image to convert to the given hue.
    hue : array-like of shape (3,)
        The RGB values of the hue to apply to the image.
    alpha : float
        Alpha level to apply to the hue.

    Returns:
    --------
    A numpy array with the same shape as image, but colorized
    appropriately.
    """
    # mask of color to super-impose
    color_mask = np.zeros_like(img).astype(np.float32)
    color_mask[:, :] = hue

    # convert to HSV colorspace
    img_hsv = color.rgb2hsv(img)
    color_hsv = color.rgb2hsv(color_mask)

    # replace hue and saturation of the original image with the mask
    img_hsv[..., 0] = color_hsv[..., 0]
    img_hsv[..., 1] = color_hsv[..., 1] * alpha

    return color.hsv2rgb(img_hsv)
