"""
Image Histogram
===============
"""
import numpy as np

import image_vis as ivs


np.random.seed(123)

image_dir, data = ivs.load_dataset('rothko')

n_samples = data.shape[0]
data['x'] = np.random.randn(n_samples)

with ivs.data_context(image_col='image_path', image_dir=image_dir):
    ivs.image_histogram(x='x', sort_by=ivs.HSVFeatures.SATURATION, data=data)
