import os

import numpy as np


class ImageDataBundle(object):
    def __init__(self, image_dir, image_files, target=None):
        self.image_dir = image_dir
        self.image_files = image_files
        self.target = target

    def __repr__(self):
        n_samples = len(self.image_files)

        n_classes = np.unique(self.target).shape[0] if self.target else 0
        return ('%s(image_dir=%s, n_samples=%d, n_classes=%d)' % (
                    self.__class__.__name__,
                    self.image_dir,
                    n_samples,
                    n_classes)
        )


def get_data_home(data_home=None):
    if data_home is None:
        data_home = os.environ.get('IMAGEVIS_DATA',
                                   os.path.join('~', 'imagevis_data'))

    data_home = os.path.expanduser(data_home)
    if not os.path.exists(data_home):
        os.makedirs(data_home)

    return data_home
