"""
Image Grid Orderd by Column
===========================
"""
import mosaic as ms

image_dir, data = ms.load_dataset('mnist')

with ms.data_context(image_col='image_path', image_dir=image_dir):
    ms.image_grid(data=data, sort_by='label')
