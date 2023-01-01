import time

import imutils
from cv2 import cv2
from keras.models import load_model

import CreateData as d


class SlidingWindow:
    def __init__(self, img_path):
        self.img = cv2.imread(img_path)
        self.img_path = img_path
        (self.win_w, self.win_h) = (64, 64)
        self.step_size = 12
        self.scale = 1.5
        self.min_size = (200,200)

    def resizeImage(self,):
        # compute the new dimensions of the image and resize it
        # if self.img.size > 2700000:
        w = int(self.img.shape[1] / 2)
        self.img = imutils.resize(self.img, width=w)
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

                window = d.shapeWindow(window)
                
                hog = d.hogFeatures(window).reshape(1,64,64,1)
                pred = model.predict(hog)
                avg_pred = (sum(pred) / len(pred))[0]
                avg_pred = (sum(avg_pred) / len(avg_pred))[0]

                print(avg_pred)

                if avg_pred > 0.7:
                    cv2.rectangle(resized, (x, y), (x + self.win_w, y + self.win_h), (255, 0, 0), 2)

                clone = resized.copy()
                cv2.rectangle(clone, (x, y), (x + self.win_w, y + self.win_h), (0, 255, 0), 2)
                cv2.imshow("Window", clone)
                cv2.waitKey(1)
                time.sleep(0.025)

if __name__ == '__main__':
    d = d.Data()
    x_train, x_test, y_train, y_test = d.shapeData()    

    model = load_model('/Users/jadenvanrijswijk/Downloads/CarPredictionAI/models/m8')

    s = SlidingWindow('/Users/jadenvanrijswijk/Downloads/CarPredictionAI/data/validation/trafficjam.jpeg')
    s.loopWindow()
