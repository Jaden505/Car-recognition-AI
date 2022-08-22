import glob, pickle
import numpy as np
from sklearn.model_selection import train_test_split
from skimage.feature import hog
from skimage import exposure

class Data:
    def shapeData(self):
        with open('data/traintest/hogimages2.pkl', 'rb') as f:
            x = pickle.load(f)
            
        x = np.stack(x, axis=0)  
            
        y = np.append(np.full(8792, 1), np.full(8968, 0))

        x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.01)


        x_train = x_train.reshape(len(x_train), 64, 64, 1)
        x_test = x_test.reshape(len(x_test), 64, 64, 1)

        return x_train, x_test, y_train, y_test

    def createDataset(self):
        data = []

        for f in glob.glob('/Users/jadenvanrijswijk/Desktop/vehicles/*'):
            for im_path in glob.glob(f'{f}/*.png'):
                hog_img = self.hogFeatures(im_path)
                data.append(hog_img)

        for f in glob.glob('/Users/jadenvanrijswijk/Desktop/nonvehicles/*'):
            for im_path in glob.glob(f'{f}/*.png'):
                hog_img = self.hogFeatures(im_path)
                data.append(hog_img)

        with open('data/traintest/hogimages3.pkl', 'wb') as f:
            pickle.dump(data, f)

    def hogFeatures(self, im):
        fd, hog_im = hog(im, orientations=8, pixels_per_cell=(8, 8), 
            cells_per_block=(2, 2), block_norm='L1', visualize=True, 
            transform_sqrt=True) 

        hog_image_rescaled = exposure.rescale_intensity(hog_im, in_range=(0.4, 8))

        
        return hog_image_rescaled

    def shapeWindow(self, im):
        im = im.tolist()

        for x in range(len(im)):
            for y in range(len(im[x])):
                im[x][y] = [int(sum(im[0][0])/3)]
        
        im = np.asarray(im)
        im = im.reshape(64,64,1)

        return im

if __name__ == '__main__':
    d = Data()
    x_train, x_test, y_train, y_test = d.shapeData()