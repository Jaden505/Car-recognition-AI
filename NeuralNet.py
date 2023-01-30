import os
from keras.layers import *
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from keras.models import Sequential

import CreateData as d

# Allow Tensorflow to train the model further
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Disables Tensorflow warning message


class NN:
    def __init__(self):
        self.model = None

    def Model(self, ):
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(512, activation='relu'),
            Dense(1, activation='sigmoid')
        ])

        self.model = model

    def Train(self, ):
        # Compile the model
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Train the model
        self.model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

        # Evaluate the model on test data
        test_loss, test_acc = self.model.evaluate(y_test, y_test, verbose=0)
        print('Test accuracy:', test_acc)

        self.model.save('models/m1')


if __name__ == '__main__':
    d = d.Data()
    x_train, x_test, y_train, y_test = d.shapeData()

    n = NN()
    n.Model()
    n.Train()
