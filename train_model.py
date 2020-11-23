###################################################
# Should only run to train model.                 #
# Download training dataset from                  #
# https://www.kaggle.com/grassknoted/asl-alphabet #
###################################################

# import required modules
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt 
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Model

# import created model
from net import Net

# Dimensions of our images
img_width, img_height = 64, 64

# 3 channel image
no_of_channels = 3

# Training variables
epochs = 6
batch_size = 64

# Train data Directory
train_data_dir = 'data/train/' 

# Load model from Net
model = Net.build(width = img_width, height = img_height, depth = no_of_channels)
print('--------building done--------')
model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
print('--------compiling done--------')

def preprocess_image(image):
    '''Function that will be implied on each input. The function
    will run after the image is resized and augmented.
    The function should take one argument: one image (Numpy tensor
    with rank 3), and should output a Numpy tensor with the same
    shape.'''
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    return sobely

train_datagen = ImageDataGenerator(
    samplewise_center=True,
    samplewise_std_normalization=True,
    preprocessing_function=preprocess_image,
    validation_split=0.1,)

# this is the augmentation configuration used for training
# horizontal_flip = False, as we need to retain signs
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True,
    subset='training',)

validation_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation')

# fit the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples / batch_size, 
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples / batch_size,  
    verbose=1,
)  

# evaluate on validation dataset
model.evaluate(validation_generator)
# save weights in a file
model.save_weights('trained_weights_2.h5') 

# Loss Curves
plt.figure(figsize=[8,6])
plt.plot(history.history['loss'],'r',linewidth=3.0)
plt.plot(history.history['val_loss'],'b',linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.title('Loss Curves',fontsize=16)
 
# Accuracy Curves
plt.figure(figsize=[8,6])
plt.plot(history.history['accuracy'],'r',linewidth=3.0)
plt.plot(history.history['val_accuracy'],'b',linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('Accuracy Curves',fontsize=16)
plt.show()
