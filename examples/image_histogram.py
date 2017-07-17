"""
Image Histogram
===============
"""
import matplotlib.pyplot as plt
import numpy as np

import image_vis

rng = np.random.RandomState(123)

with image_vis.image_dir('../image_vis/datasets/rothko_images'):
    data = image_vis.image_io.directory_to_dataframe()
    data['x'] = rng.randn(data.shape[0])
    histo = image_vis.image_histogram(
        'image_path', x='x', y=image_vis.HSVFeatures.SATURATION, data=data)

f, ax = plt.subplots(figsize=(10, 10))
fig = plt.imshow(histo)
plt.axis('off')
