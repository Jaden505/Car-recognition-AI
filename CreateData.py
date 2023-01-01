import glob, pickle
import numpy as np
from sklearn.model_selection import train_test_split
from skimage.feature import hog
from skimage.io import imread
from skimage import exposure
from PIL import Image
import os

class Data:
    def __init__(self):
        self.VECHILES_PATH = '/Users/jadenvanrijswijk/Desktop/CarRecognitionData/vehicles/'
        self.NONVECHILES_PATH = '/Users/jadenvanrijswijk/Desktop/CarRecognitionData/non-vehicles/'
        self.GRAYVECHILES = 'data/vechiles/GrayVechilesDump.pkl'

        self.VECHILES_DATA_SIZE = sum([len(files) for r, d, files in os.walk(self.VECHILES_PATH)])-1
        self.NONVECHILES_DATA_SIZE = sum([len(files) for r, d, files in os.walk(self.NONVECHILES_PATH)])-1

    def shapeData(self):
        with open(self.GRAYVECHILES, 'rb') as f:
            x = pickle.load(f)
            
        x = np.stack(x, axis=0)  
            
        y = np.append(np.full(self.VECHILES_DATA_SIZE, 1), np.full(self.NONVECHILES_DATA_SIZE, 0))

        x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.01)


        x_train = x_train.reshape(len(x_train), 64, 64, 1)
        x_test = x_test.reshape(len(x_test), 64, 64, 1)

        return x_train, x_test, y_train, y_test

    def createDataset(self):
        data = []

        for f in glob.glob(self.VECHILES_PATH + '*'):
            for im_path in glob.glob(f'{f}/*.png'):
                data.append(self.shapeImage(im_path))

        for f in glob.glob(self.NONVECHILES_PATH + '*'):
            for im_path in glob.glob(f'{f}/*.png'):
                data.append(self.shapeImage(im_path))

        with open(self.GRAYVECHILES, 'wb') as f:
            pickle.dump(data, f)

    def shapeImage(self, im_path):
       im = Image.open(im_path).convert('L')
       im = im.resize((64, 64))
       return np.array(im)

if __name__ == '__main__':
    d = Data()
    # d.createDataset()
    x_train, x_test, y_train, y_test = d.shapeData()
    print(x_test, y_test)