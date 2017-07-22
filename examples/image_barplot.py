"""
Mosaic Plot Orderd by Column
============================
"""
import pandas as pd
import image_vis as ivs


image_dir, data = ivs.load_dataset('mnist')
data = pd.read_csv('cifar10_embedding_data.csv')

with ivs.data_context(image_col='image_path', image_dir='../image_vis/datasets/cifar10'):
    ax = ivs.image_barplot(y='label', data=data, figsize=(15, 5))
