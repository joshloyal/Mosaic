"""
Scatter Plot
============
"""
import mosaic as ms

image_dir, data = ms.load_dataset('mnist')

with ms.data_context(image_col='image_path', image_dir=image_dir):
    ms.scatter_plot(x='x', y='y', data=data, hue='label', figsize=(15, 15))
