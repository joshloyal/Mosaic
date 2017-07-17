import contextlib

_IMAGE_DIR = ''


@contextlib.contextmanager
def image_dir(image_dir):
    global _IMAGE_DIR
    _IMAGE_DIR = image_dir
    yield image_dir
    _IMAGE_DIR = ''


def get_image_dir():
    global _IMAGE_DIR
    return _IMAGE_DIR
