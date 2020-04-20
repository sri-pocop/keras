
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
img = load_img('keras/image/Koala1.tif')  #image path
x = img_to_array(img)
x = x.reshape((1,) + x.shape)

i = 1
for batch in datagen.flow(x, batch_size = 1,
                           save_to_dir = 'keras/agmented', # image destination path
                           save_prefix = 'koala',
                           save_format='tif'):
    
    i += 1
    if i > 10: # number of augmented images required
        break
