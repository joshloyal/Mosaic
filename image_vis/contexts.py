import contextlib


class PlottingContext(object):
    def __init__(self):
        self._image_col = None
        self._image_dir = ''

    def set_context(self, image_col=None, image_dir=''):
        self._image_col = image_col
        self._image_dir = image_dir

    def unset_context(self):
        self._image_col = None
        self._image_dir = ''

    @property
    def image_dir(self):
        return self._image_dir

    @property
    def image_col(self):
        return self._image_col


_PLOTTING_CONTEXT = PlottingContext()


@contextlib.contextmanager
def plotting_context(image_col=None, image_dir=''):
    global _PLOTTING_CONTEXT
    _PLOTTING_CONTEXT.set_context(image_col=image_col, image_dir=image_dir)
    yield _PLOTTING_CONTEXT
    _PLOTTING_CONTEXT.unset_context()


def get_plotting_context():
    global _PLOTTING_CONTEXT
    return _PLOTTING_CONTEXT


def get_image_dir():
    return get_plotting_context().image_dir


def get_image_col():
    return get_plotting_context().image_col
