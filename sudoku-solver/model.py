import seaborn as sns
import keras

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Lambda, MaxPooling2D # convolution layers
from tensorflow.keras.layers import Dense, Dropout, Flatten # core layers
from keras.preprocessing import image
from keras.layers.normalization import BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.datasets import mnist
from keras.utils import np_utils


def train():
    (X_train,y_train),(X_test,y_test)=mnist.load_data()
    X_train = X_train.reshape(60000, 28, 28, 1)
    X_test = X_test.reshape(10000, 28, 28, 1)
    X_train=X_train/255
    X_test = X_test / 255

    y_train=np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    num_classes=y_test.shape[1]

    ## Declare the model
    model = Sequential()

    ## Declare the layers
    layer_1 = Conv2D(32, kernel_size=3, activation='relu', input_shape=(28, 28, 1))
    layer_2 = Conv2D(64, kernel_size=3, activation='relu')
    layer_3 = Flatten()
    layer_4 = Dense(10, activation='softmax')

    ## Add the layers to the model
    model.add(layer_1)
    model.add(layer_2)
    model.add(layer_3)
    model.add(layer_4)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5)
    model.save('keras_digit_model.h5')
    return model

model = train()


