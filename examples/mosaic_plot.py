"""
Mosaic Plot
===========
"""
import image_vis


with image_vis.image_dir('../image_vis/datasets/rothko_images'):
    data = image_vis.image_io.directory_to_dataframe()
    mosaic = image_vis.mosaic_plot(
        'image_path', data=data, sort_by=image_vis.HSVFeatures.SATURATION)
