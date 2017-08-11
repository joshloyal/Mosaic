"""
Scatter Grid
=============
"""
import mosaic as ms

image_dir, data = ms.load_dataset('mnist')

with ms.data_context(image_col='image_path', image_dir=image_dir):
    ms.scatter_grid(x='x', y='y', data=data, hue='label', n_jobs=-1)
