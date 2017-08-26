from mosaic.datasets.rothko import fetch_rothko_images
from mosaic.datasets.mnist import fetch_mnist_images
from mosaic.datasets.cifar10 import fetch_cifar10_images
from mosaic.datasets.fashion import fetch_fashion_images


def load_dataset(name, kind='train'):
    if name == 'rothko':
        return fetch_rothko_images()
    elif name == 'mnist':
        return fetch_mnist_images()
    elif name == 'cifar10':
        return fetch_cifar10_images()
    elif name == 'fashion':
        return fetch_fashion_images(kind=kind)
    raise ValueError('Unknown dataset {}'.format(name))
