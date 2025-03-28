# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 21:23:11 2022

@author: Ameer
"""

from keras.models import load_model
#import cv2
import numpy as np
from tensorflow import keras
from keras import metrics

from tensorflow.keras import optimizers

from keras.preprocessing import image
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input


class_names = np.array(['cough','dog']) 

def top_5_accuracy(y_true, y_pred):
    return metrics.top_k_categorical_accuracy(y_true, y_pred, k=5)

dependencies = {
    'top_5_accuracy': top_5_accuracy
}

pre_trained_model = load_model('./cough_model_artifacts/esc50_vgg16_stft_weights_train_last_2_base_layers_best.hdf5', custom_objects=dependencies)

pre_trained_model.compile(loss='categorical_crossentropy',
              optimizer='SGD',
              metrics=['accuracy',top_5_accuracy])

img_width, img_height = 224, 224

input_tensor = Input(shape=(224,224,3))

validation_data_dir = './wavelets/v1-dataset/samples/'

validation_datagen = image.ImageDataGenerator(
    rescale=1./255)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    )



prediction_classes = np.argmax(pre_trained_model.predict(validation_generator), axis = -1)

print(validation_generator.filenames)
print(pre_trained_model.predict(validation_generator))
#print(prediction_classes)
names = [class_names[i] for i in prediction_classes]
print (names)