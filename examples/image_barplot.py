"""
Image BarPlot
=============
"""
import mosaic as ms


images, labels = ms.load_dataset('cifar10')

images = images[:1000]
labels = labels[:1000]
ms.image_barplot(y=labels, images=images, bar_height=50)
