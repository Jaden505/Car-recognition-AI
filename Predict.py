import CreateData as d

from keras.models import load_model


class Predict:
    def __init__(self):
        self.model = load_model('models/m1')

    def singleCarImagePredict(self, img_path):
        shaped_img = d.shapeImage(img_path)
        shaped_img = shaped_img.reshape(1, 64, 64, 1)

        prediction = self.model.predict(shaped_img)

        print(prediction)

        if (prediction[0][0] > 0.5):
            print('Car')
        else:
            print('Not a car')


if __name__ == "__main__":
    d = d.Data()
    p = Predict()

    p.singleCarImagePredict('data/validation/carFromHighway.png')
    # p.singleCarImagePredict('data/validation/nocar.png')
