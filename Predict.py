import CreateData as d

from keras.models import load_model

class Predict:
    def __init__(self):
        self.model = load_model('models/m1')

    def singleCarImagePredict(self, img_path):
        shaped_img = d.shapeImage(img_path)

        print(shaped_img)

        self.model.predict(shaped_img)

if __name__ == "__main__":
    d = d.Data()
    p = Predict()

    p.singleCarImagePredict('data/validation/car.jpeg')