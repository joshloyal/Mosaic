"""
Image BarPlot
=============
"""
import image_vis as ivs


images, labels = ivs.load_dataset('cifar10')

images = images[:1000]
labels = labels[:1000]
ivs.image_barplot(y=labels, images=images, bar_height=50)
