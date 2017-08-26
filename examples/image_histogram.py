"""
Image Histogram
===============
"""
import numpy as np

import mosaic as ms

np.random.seed(123)

image_dir, data = ms.load_dataset('rothko')

n_samples = data.shape[0]
data['x'] = np.random.randn(n_samples) + 10

with ms.data_context(image_col='image_path', image_dir=image_dir):
    ms.image_histogram(x='x', sort_by=ms.SATURATION, data=data, image_size=50)
