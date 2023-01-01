import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
from keras.layers import *
from keras import backend as K
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model

import CreateData as d

# Allow Tensorflow to train the model further 
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Disables Tensorflow warning message

class NN:
    def Model(self,):
        model = Sequential()

        # Input image array
        model.add(Input(shape=(64,64,1)))

        # Hidden layers
        # model.add(Conv2D(filters=8,kernel_size=(3,3),padding="same", activation="relu"))
        # model.add(Conv2D(filters=10,kernel_size=(3,3),padding="same", activation="relu"))
        # model.add(Conv2D(filters=10,kernel_size=(3,3),padding="same", activation="relu"))
        # model.add(MaxPool2D(pool_size=(10,10),strides=(2,2)))
        # model.add(Conv2D(filters=12,kernel_size=(3,3),padding="same", activation="relu"))
        # model.add(Conv2D(filters=12,kernel_size=(3,3),padding="same", activation="relu"))
        # model.add(Conv2D(filters=12,kernel_size=(3,3),padding="same", activation="relu"))
        # model.add(MaxPool2D(pool_size=(15,15),strides=(2,2)))
        # model.add(Conv2D(filters=16,kernel_size=(3,3),padding="same", activation="relu"))
        # model.add(Conv2D(filters=16,kernel_size=(3,3),padding="same", activation="relu"))
        # model.add(Conv2D(filters=16,kernel_size=(3,3),padding="same", activation="relu"))
        model.add(Flatten())

        # Output 0 or 1 (true or false) based on if there is a car
        model.add(Dense(1, activation='sigmoid'))
        
        self.model = model

    def Train(self,):
        opt = SGD(learning_rate=0.01, momentum=0.9)

        self.model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

        self.model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100, verbose=2,
         steps_per_epoch=800, validation_steps=10, batch_size=8)

        self.model.save('models/m1')
                
    def testModelAccuracy(self,):
        correct = 0
        incorrect = 0

        self.model = load_model('/Users/jadenvanrijswijk/Downloads/CarPredictionAI/models/m1')

        for im, answer in zip(x_test, y_test):
            im = im.reshape(1,64,64,1)

            pred = self.model.predict(im)
            avg_pred = sum(pred) / len(pred)
            avg_pred = sum(avg_pred) / len(avg_pred)

            if avg_pred > 0.9 and answer == 1:
                correct += 1
            elif avg_pred < 0.9 and answer == 0:
                correct += 1
            else:
                incorrect += 1

        print('Total corrects: ', correct)
        print('Total incorrects: ', incorrect)
            
if __name__ == '__main__':
    d = d.Data()
    x_train, x_test, y_train, y_test = d.shapeData()

    n = NN()
    n.Model()
    n.Train()
    n.testModelAccuracy()
    