"""
Mosaic Plot with Custom Ordering
================================
"""
import pandas as pd

import image_vis as ivs


data = pd.read_csv('mnist_data.csv')

context = dict(
    image_col='image_path',
    image_dir='../image_vis/datasets/mnist'
)

with ivs.plotting_context(**context):
    ivs.mosaic_plot(data=data, sort_by='label', n_samples=625)
