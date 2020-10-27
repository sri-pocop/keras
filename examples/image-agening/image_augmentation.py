# -*- coding: utf-8 -*-
"""
Created on Wed May 20 13:02:06 2020

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
img = load_img('D:\python\data\image\dog1.jpg')
x = img_to_array(img)
x = x.reshape((1,) + x.shape)

file_name = 'dog_' #filename prefix
i = 1
format_type = 'jpg' # format type for generated image
save_dir = 'D:/python/CNN/test_data/dog/'
for batch in datagen.flow(x, batch_size = 1,
                           save_to_dir = save_dir, save_prefix = 'dog',
                           save_format = format_type):
    i += 1
    if i > 70:
        break

import os
os.getcwd()
collection = save_dir
for i, filename in enumerate(os.listdir(collection)):
    i+=1
    os.rename(save_dir + filename, save_dir + file_name + str(i) + "."+format_type)