"""
Scatter Plot
===============
"""
import numpy as np

import image_vis as ivs


rng = np.random.RandomState(123)

image_dir, data = ivs.load_dataset('rothko')

with ivs.plotting_context(image_col='image_path', image_dir=image_dir):
    data['x'] = rng.randn(data.shape[0])
    data['y'] = rng.randn(data.shape[0])
    ivs.scatter_plot(x='x', y='y', data=data)
