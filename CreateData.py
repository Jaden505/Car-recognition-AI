import NeuralNet as n
import Predict as s

import glob, pickle, imutils
from cv2 import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from skimage.feature import hog
from skimage import exposure

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
        # y_train = y_train.reshape(len(y_train), 1,1,1)

        return x_train, x_test, y_train, y_test

    def createDataset(self, x, v):
        for f in glob.glob(f'/Users/jadenvanrijswijk/Desktop/{v}/*'):
            for im_path in glob.glob(f'{f}/*.png'):
                hog_img = self.hogFeatures(im_path)
                x.append(hog_img)

        return x

    def hogFeatures(self, im):
        # im = cv2.cvtColor(im, cv2.COLOR_BGR2YCR_CB)

        fd, hog_im = hog(im, orientations=8, pixels_per_cell=(8, 8), 
            cells_per_block=(2, 2), block_norm='L1', visualize=True, 
            transform_sqrt=True) 

        hog_image_rescaled = exposure.rescale_intensity(hog_im, in_range=(0.4, 8))

        # cv2.imshow('cars', hog_image_rescaled)
        # cv2.imshow('cars2', im)
        # cv2.waitKey(0)

        # hog_image_rescaled.reshape(64,64,1)
        
        return hog_image_rescaled

if __name__ == '__main__':
    d = Data()
    x_train, x_test, y_train, y_test = d.shapeData()
