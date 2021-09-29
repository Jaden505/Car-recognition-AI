from keras import layers, Input, models, optimizers
from keras.layers import *
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import glob, pickle, os, imutils, time
from cv2 import cv2
from skimage.feature import hog
from matplotlib import pyplot as plt
from skimage import exposure
from sklearn.utils import shuffle
from tensorflow import keras
import matplotlib.patches as patches
from PIL import Image
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class Data:
    def shapeData(self,):
        with open('data/traintest/hogimages.pkl', 'rb') as f:
            x = pickle.load(f)
            
        x = np.stack(x, axis=0)  
            
        y = np.append(np.full(8792, 1), np.full(8968, 0))

        x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.01)

        x_train = x_train.reshape(len(x_train), 64, 64, 1)
        x_test = x_test.reshape(len(x_test), 64, 64, 1)
        # y_train = y_train.reshape(len(y_train), 1)

        return x_train, x_test, y_train, y_test

    def createDataset(self, x, v):
        for f in glob.glob(f'/Users/jadenvanrijswijk/Desktop/{v}/*'):
            for im_path in glob.glob(f'{f}/*.png'):
                hog_img = self.hogFeatures(im_path)
                x.append(hog_img)

        return x

    def hogFeatures(self, im):
        im = cv2.cvtColor(im, cv2.COLOR_BGR2YCR_CB)

        fd, hog_im = hog(im, orientations=8, pixels_per_cell=(8, 8), 
            cells_per_block=(2, 2), block_norm='L1', visualize=True, 
            transform_sqrt=True) 

        hog_image_rescaled = exposure.rescale_intensity(hog_im, in_range=(0.4, 8))

        # cv2.imshow('cars', hog_image_rescaled)
        # cv2.waitKey(0)
        
        return hog_image_rescaled

class NN:
    def Model(self,):
        # self.model = models.Sequential()
        # self.model.add(Input(shape=(64,)))
        # self.model.add(layers.Dense(256, activation="relu"))
        # self.model.add(layers.Dense(128, activation="relu"))
        # self.model.add(layers.Dense(1, activation="sigmoid"))

        model = Sequential()

        model.add(Conv2D(input_shape=(64,64,1),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
        model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
        model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
        model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
        model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
        model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(3,3),strides=(3,3)))

        self.model = model

    def Train(self,):
        # opt = SGD(lr=0.01, momentum=0.9)
        # self.model.compile(loss='mean_squared_logarithmic_error', optimizer=opt, metrics=['mse'])
        # history = self.model.fit(x_train, y_train, validation_data=(x_test, y_test),
        #                     epochs=60, verbose=1, batch_size=30, shuffle=True)


        model_path = '/Users/jadenvanrijswijk/Downloads/CarPredictionAI/models/m6'
        opt = Adam(learning_rate=0.001)

        self.model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
        checkpoint = ModelCheckpoint("%s/vgg16_1.h5" % model_path, monitor='loss', verbose=1, save_best_only=True, 
                                        save_weights_only=False, mode='auto', save_freq=1)
        early = EarlyStopping(monitor='loss', min_delta=0, patience=20, verbose=1, mode='auto')
        hist = self.model.fit(x_train, y_train, validation_data=(x_test, y_test), steps_per_epoch=100,
                            validation_steps=10, epochs=100, callbacks=[checkpoint,early])

        self.model.save(model_path)

    def testModelAccuracy(self,):
        correct = 0
        incorrect = 0

        for im, answer in zip(x_test, y_test):
            pred = model.predict(im)
            avg_pred = (sum(pred) / len(pred))[0]

            if avg_pred > 0.9 and answer == 1 :
                correct += 1
            elif avg_pred < 0.9 and answer == 0:
                correct += 1
            else:
                incorrect += 1

        print('Total corrects: ', correct)
        print('Total incorrects: ', incorrect)

class SlidingWindow:
    def __init__(self, img_path):
        self.img = cv2.imread(img_path)
        self.img_path = img_path
        (self.win_w, self.win_h) = (64, 64)
        self.step_size = int(self.img.size / 100000)
        self.scale = 1.5
        self.min_size = (200,200)

    def resizeImage(self,):
        # compute the new dimensions of the image and resize it
        # if self.img.size > 2700000:
        #     w = int(self.img.shape[1] / self.scale)
        #     self.img = imutils.resize(self.img, width=w)
        #     self.step_size = int(self.img.size / 100000)

        yield self.img
        
        while True:
            w = int(self.img.shape[1] / self.scale)
            self.img = imutils.resize(self.img, width=w)

            if self.img.shape[0] < self.min_size[1] or self.img.shape[1] < self.min_size[0]:
                break
                
            # self.step_size = int(self.img.size / 100000)

            yield self.img

    def moveWindow(self,):
        # slide a window across the image
        for x in range(0, self.img.shape[0], self.step_size):
            for y in range(0, self.img.shape[1], self.step_size):
                yield (x, y, self.img[x:x + self.win_w, y:y + self.win_h])

    def loopWindow(self,):
        # loop over the sliding window for each resized image
        for resized in self.resizeImage():
            for (x, y, window) in self.moveWindow():
                if window.shape[0] != self.win_h or window.shape[1] != self.win_h:
                    continue

                hog = d.hogFeatures(window)
                pred = model.predict(hog)
                avg_pred = (sum(pred) / len(pred))[0]

                print(avg_pred)

                if avg_pred > 0.7:
                    cv2.rectangle(resized, (x, y), (x + self.win_w, y + self.win_h), (255, 0, 0), 2)

                clone = resized.copy()
                cv2.rectangle(clone, (x, y), (x + self.win_w, y + self.win_h), (0, 255, 0), 2)
                cv2.imshow("Window", clone)
                cv2.waitKey(1)
                time.sleep(0.025)
            
if __name__ == '__main__':
    d = Data()
    x_train, x_test, y_train, y_test = d.shapeData()
    
    # n = NN()
    # n.Model()
    # n.Train()
    
    model = models.load_model('/Users/jadenvanrijswijk/Downloads/CarPredictionAI/models/m6/vgg16_1.h5')

    im = cv2.imread('/Users/jadenvanrijswijk/Downloads/CarPredictionAI/data/validation/car.jpeg')

    im.reshape(64,64,1)

    hog = d.hogFeatures(im)
    print(hog.shape, hog.size)
    pred = model.predict(hog)
    avg_pred = (sum(pred) / len(pred))[0]

    print(avg_pred)

    # s = SlidingWindow('/Users/jadenvanrijswijk/Downloads/CarPredictionAI/data/validation/trafficjam.jpeg')
    # s.loopWindow()
    
# loss: 0.0654 - mse: 0.1396 - val_loss: 0.0716 - val_mse: 0.1495