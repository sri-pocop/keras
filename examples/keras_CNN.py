# -*- coding: utf-8 -*-
"""
Created on Wed May 20 13:24:55 2020

@author: srinivasan.c
"""
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
from keras.preprocessing import image
import os, sys

classifier = Sequential()
classifier.add(Conv2D(32,(3,3), input_shape = (64, 64, 3), activation= 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))
classifier.add(Flatten())
classifier.add(Dense(units = 128 , activation = 'relu'))
classifier.add(Dense(units = 1 , activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

train_datagen = ImageDataGenerator( rescale = 1./255,
                                    shear_range = .2,
                                    zoom_range = .2,
                                    horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('D:/python/CNN/train_data/',
                                                target_size = (64,64),
                                                batch_size = 32,
                                                class_mode = 'binary')
test_set = test_datagen.flow_from_directory('D:/python/CNN/test_data/',
                                                target_size = (64,64),
                                                batch_size = 32,
                                                class_mode = 'binary')
class_names = os.listdir( 'D:/python/CNN/train_data/' )
classifier.fit_generator(training_set,
                            samples_per_epoch = 50,
                            nb_epoch = 110,
                            validation_data = test_set#,nb_val_sample = 147
                            )

classifier.save('dog_cat_model.h5')
model = load_model('dog_cat_model.h5')
def predict_model_type(model, path):
    """
    

    Parameters
    ----------
    model : TYPE
        pass model which to be predicted.
    path : TYPE
        path of the image file.

    Returns
    -------
    returns the image class.

    """
    image_path = path
    img = image.load_img(image_path, target_size=(64,64))
    plt.imshow(img)
    img = np.expand_dims(img, axis=0)
    #result = model.predict(img)
    result = model.predict_classes(img)
    print (result)
    #return class_names[result[0][0]]

predicted_class = predict_model_type(model,'D:/python/CNN/pred/test.jpg')
print(predicted_class)