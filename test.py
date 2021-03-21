# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 00:05:24 2021

@author: M
"""

import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model

length, height = 100, 100
modelo = './model/model.h5'
pesos = './model/pesos.h5'

cnn = load_model(modelo)
cnn.load_weights(pesos)

def predict(file):
    x = load_img(file, target_size = (length, height))
    x = img_to_array(x)
    x = np.expand_dims(x, axis = 0)
    arreglo = cnn.predict(x)
    result = arreglo[0]
    type_coin = np.argmax(result)
    
    if type_coin == 0:
        print('Moneda no registrada')
    elif type_coin == 1:
        print('Un nuevo sol')
    return type_coin

predict('Sol1.jpg')