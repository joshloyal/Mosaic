"""
Distance Grid
=============
"""
import image_vis as ivs

image_dir, data = ivs.load_dataset('mnist')

with ivs.plotting_context(image_col='image_path', image_dir=image_dir):
    ivs.distance_grid(x='x', y='y', data=data, hue='label', n_jobs=-1)
