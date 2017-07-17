"""
Mosaic Plot
===========
"""
import matplotlib.pyplot as plt

import image_vis


with image_vis.image_dir('../image_vis/datasets/rothko_images'):
    data = image_vis.image_io.directory_to_dataframe()
    mosaic = image_vis.mosaic_plot(
        'image_path', data=data, sort_by=image_vis.HSVFeatures.SATURATION)

f, ax = plt.subplots(figsize=(10, 10))
fig = plt.imshow(mosaic)
plt.axis('off')
