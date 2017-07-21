"""
Scatter Plot
============
"""
import image_vis as ivs

image_dir, data = ivs.load_dataset('mnist')

with ivs.data_context(image_col='image_path', image_dir=image_dir):
    ivs.scatter_plot(x='x', y='y', data=data, hue='label', figsize=(15, 15))
