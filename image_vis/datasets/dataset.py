from image_vis.datasets.rothko import fetch_rothko_images
from image_vis.datasets.mnist import fetch_mnist_images
from image_vis.datasets.cifar10 import fetch_cifar10_images


def load_dataset(name):
    if name == 'rothko':
        return fetch_rothko_images()
    elif name == 'mnist':
        return fetch_mnist_images()
    elif name == 'cifar10':
        return fetch_cifar10_images()
    raise ValueError('Unknown dataset {}'.format(name))
