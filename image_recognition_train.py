#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 29 12:54:23 2021

@author: nazish
"""
from tensorflow.keras.datasets import mnist
import matplotlib. pyplot as plt
import numpy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.python.keras.layers.convolutional import Conv2D, MaxPooling2D
from tensorflow.python.keras.utils import np_utils
from tensorflow.keras import backend as K

# # For Genrating test images
# from PIL import Image
# from tensorflow.keras.datasets import mnist
# import numpy as np

# (X_train, y_train), (X_test, y_test) = mnist.load_data()
# for i in np.random.randint(0, 10000+1, 10):
#     arr2im = Image.fromarray(X_train[i])
#     arr2im.save('test_images/{}.png'.format(i), "PNG")


# Load dataset (download if needed)
(X_train, y_train), (X_test, y_test) = mnist.load_data()

plt.subplot(221)
plt.imshow(X_train[0], cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(X_train[1], cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(X_train[2], cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(X_train[3], cmap=plt.get_cmap('gray'))

plt.show()


K.set_image_data_format('channels_last')

# fix the seed 
seed = 7
numpy.random.seed(seed)

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')

X_train = X_train / 255
X_test = X_test / 255

# one hot encoding
# output - [ 0 0 0 0 0 1 0 0 0 0 ]
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

num_classes = y_train.shape[1]

def baseline_model():
    model = Sequential() #as we have static NN
    model.add(Conv2D(8, (3,3), input_shape=(28, 28, 1), activation='relu')) # 8 filters of 3x3 conv
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Flatten())
    model.add(Dense(4, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                 metrics=['accuracy'])
    
    return model

# build a model
model = baseline_model()

# Fit 
#print(X_test.shape)
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3,
          batch_size=32, verbose=2)

model.save('model.h5')

# Final evaluation
scores = model.evaluate(X_test, y_test, verbose=0)
print("CNN error: %.2f%%" % (100 - scores[1]*100))
