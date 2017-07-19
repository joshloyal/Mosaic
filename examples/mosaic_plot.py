"""
Mosaic Plot
===========
"""
import image_vis as ivs


image_dir, data = ivs.load_dataset('rothko')

with ivs.plotting_context(image_col='image_path', image_dir=image_dir):
    ivs.mosaic_plot(data=data, sort_by=ivs.HSVFeatures.SATURATION)
