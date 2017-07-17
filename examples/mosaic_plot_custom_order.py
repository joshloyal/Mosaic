"""
Mosaic Plot with Custom Ordering
================================
"""
import pandas as pd

import image_vis


with image_vis.image_dir('../image_vis/datasets/mnist'):
    data = pd.read_csv('mnist_data.csv')
    mosaic = image_vis.mosaic_plot(
        'image_path', data=data, sort_by='label', n_samples=625)
