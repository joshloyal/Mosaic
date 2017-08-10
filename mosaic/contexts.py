import contextlib


class DataContext(object):
    """Wrapper around the data context variables to set
    the data context temporarly.
    """
    def __init__(self):
        self._image_col = None
        self._image_dir = ''
        self._image_data = None

    def set_context(self, image_col=None, image_dir='', image_data=None):
        self._image_col = image_col
        self._image_dir = image_dir
        self._image_data = image_data

    def unset_context(self):
        self._image_col = None
        self._image_dir = ''
        self._image_data = None

    @property
    def image_dir(self):
        return self._image_dir

    @property
    def image_col(self):
        return self._image_col

    @property
    def image_data(self):
        return self._image_data


_DATA_CONTEXT = DataContext()


@contextlib.contextmanager
def data_context(image_col='', image_data=None, image_dir=''):
    """Set the data context parameters.

    This affects the loading of image data when constructing plots.
    The context sets the directory where images are loaded from as well as
    the name of the variable indicating the image file paths.

    Parameters
    ----------
    image_col : str
        The name of the variable indicating the image file paths
        within `data`.

    image_dir : str, optional
        The location of the image files on disk. Images will
        be loaded from files matching the pattern
        'image_dir + os.path.sep + image_path'.
        The default assumes the files are located in the current directory.

    Examples
    --------
    >>> import mosaic as ms
    >>> with ms.data_context(image_col='image_path', image_dir='/path/to/images'):
    >>>    ms.image_grid(data=data)

    """
    global _DATA_CONTEXT
    _DATA_CONTEXT.set_context(
        image_col=image_col, image_dir=image_dir, image_data=image_data)
    yield _DATA_CONTEXT
    _DATA_CONTEXT.unset_context()


def get_data_context():
    """Return the global DataContext object."""
    global _DATA_CONTEXT
    return _DATA_CONTEXT


def get_image_dir():
    """Return the `image_dir` set in the current context."""
    return get_data_context().image_dir


def get_image_col():
    """Return the `image_col` set in the current context."""
    return get_data_context().image_col


def get_image_data():
    """Return the `image_data` set in the current context."""
    return get_data_context().image_data
