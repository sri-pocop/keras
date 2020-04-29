# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 10:21:35 2020

@author: srinivasan.c
"""

from keras.preprocessing.image import ImageDataGenerator, array_to_img,img_to_array,load_img
import tensorflow as tf

datagen = ImageDataGenerator(
        rotation_range=180,
        width_shift_range = 0.2,
        height_shift_range = 0.2,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True,
        fill_mode = 'nearest'
        )
img = load_img('keras/image/Koala1.tif')
x = img_to_array(img)
x = x.reshape((1,) + x.shape)

file_name = 'sample_' #filename prefix
i = 1
format_type = 'tif' # format type for generated image
for batch in datagen.flow(x, batch_size = 1,
                           save_to_dir = 'keras/agmented', save_prefix = 'kolar',
                           save_format = format_type):
    i += 1
    if i > 10:
        break

import os
os.getcwd()
collection = "keras/agmented"
for i, filename in enumerate(os.listdir(collection)):
    i+=1
    os.rename("keras/agmented/" + filename, "keras/agmented/" + file_name + str(i) + "."+format_type)
