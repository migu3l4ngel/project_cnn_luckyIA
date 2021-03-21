# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 00:03:13 2021

@author: M
"""

import sys
import os
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation
from tensorflow.python.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.python.keras import backend as K

K.clear_session() #cerrar sesiones keras en ejecucion vigente

#variables almacenan el dir para el entrenamiento y test
data_train = './data/train' 
data_test = './data/test'

epochs = 15
height, length = 100, 100
batch_size = 8
steps = 25 
steps_test = 200
filtrosConv1= 32
filtrosConv2= 64
tam_filtro1 = (3, 3)
tam_filtro2 = (2, 2)
tam_pool = (2, 2)
clases = 2
learning_rate = 0.0005

train_datagen = ImageDataGenerator (
    rescale = 1./255,
    shear_range = 0.3,
    zoom_range = 0.3,
    horizontal_flip = True
)

test_datagen = ImageDataGenerator (
    rescale = 1./255
)
img_train = train_datagen.flow_from_directory (
    data_train,
    target_size = (height, length),
    batch_size = batch_size,
    class_mode = 'categorical'
)
img_test = test_datagen.flow_from_directory (
    data_test,
    target_size = (height, length),
    batch_size = batch_size,
    class_mode = 'categorical'
)


cnn = Sequential()

cnn.add(Convolution2D(filtrosConv1, tam_filtro1, padding = 'same', input_shape = (height, length, 3), activation = 'relu'))
cnn.add(MaxPooling2D(pool_size = tam_pool))

cnn.add(Convolution2D(filtrosConv2, tam_filtro2, padding = 'same', activation = 'relu'))
cnn.add(MaxPooling2D(pool_size = tam_pool))

cnn.add(Flatten())
cnn.add(Dense(256, activation = 'relu'))
cnn.add(Dropout(0.5))
cnn.add(Dense(clases, activation = 'softmax'))


opt = SGD(lr = learning_rate, momentum = 0.9)
cnn.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics = ['accuracy'])

cnn.fit_generator(img_train, steps_per_epoch = steps, epochs = epochs, validation_data = img_test, validation_steps = steps_test)

dir = './model/'

if not os.path.exists(dir):
    os.mkdir(dir)
cnn.save('./model/model.h5')
cnn.save_weights('./model/pesos.h5')