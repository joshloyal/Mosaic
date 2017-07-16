"""
Mosaic Plot
===========
"""
import matplotlib.pyplot as plt

import image_vis

image_dir = '../image_vis/datasets/rothko_images'
data = image_vis.image_io.directory_to_dataframe(image_dir)

mosaic = image_vis.mosaic_plot(
    'image_path',
    data=data,
    image_dir=image_dir,
    target_size=(100, 100),
    sort_by=image_vis.features.HSVFeatures.SATURATION,
    n_jobs=-1)


f, ax = plt.subplots(figsize=(10, 10))
fig = plt.imshow(mosaic)
plt.axis('off')
