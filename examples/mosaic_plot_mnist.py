"""
Mosaic Plot Orderd by Column
============================
"""
import image_vis as ivs


image_dir, data = ivs.load_dataset('mnist')

with ivs.data_context(image_col='image_path', image_dir=image_dir):
    ivs.mosaic_plot(data=data, sort_by='label')
