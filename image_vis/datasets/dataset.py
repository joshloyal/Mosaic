from image_vis.datasets.rothko import fetch_rothko_images
from image_vis.datasets.mnist import fetch_mnist_images


def load_dataset(name):
    if name == 'rothko':
        return fetch_rothko_images()
    elif name == 'mnist':
        return fetch_mnist_images()
    raise ValueError('Unknown dataset {}'.format(name))
