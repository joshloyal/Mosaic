"""
Image Histogram
===============
"""
import numpy as np

import image_vis as ivs


rng = np.random.RandomState(123)

image_dir, data = ivs.load_dataset('rothko')

with ivs.plotting_context(image_col='image_path', image_dir=image_dir):
    data['x'] = rng.randn(data.shape[0])
    ivs.image_histogram(x='x', y=ivs.HSVFeatures.SATURATION, data=data)
