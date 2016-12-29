import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU, Activation
from keras.layers.convolutional import Convolution2D
from keras.callbacks import EarlyStopping
from keras.utils.io_utils import HDF5Matrix
import h5py
import json
import gc


# Batch generator generates batch data from training h5 file.
# The h5 file & its contents are built by ./image_angle_to_h5.py from data generated in simulator training mode.
def batch_generator(path, X, y, batch_size=32):
        nb_data = HDF5Matrix(path, X).shape[0]
        i = 0
        f = h5py.File(path)
        X_train = f[X]
        y_train = f[y]
        while True:
            start = (batch_size * i) % nb_data
            end = batch_size * (i + 1) % nb_data
            i += 1 
            if end < start:
                continue
            yield (X_train[start:end], y_train[start:end])


# define CNN model
def get_model():
    row, col, ch = 66, 200, 3

    model = Sequential()
    # normalize input image.
    model.add(Lambda(lambda x: x/127.5 - 1.,
                     input_shape=(row, col, ch), output_shape=(row, col, ch)))

    # convolution layers
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode="valid"))
    model.add(Activation('relu'))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode="valid"))
    model.add(Activation('relu'))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode="valid"))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, border_mode="valid"))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, border_mode="valid"))

    # flatten CNN layer for fc layers
    model.add(Flatten())
    # dropout for regularization usage
    model.add(Dropout(.2))
    model.add(ELU())

    # fc layers for regression
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dense(50))
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Dropout(.5))
    model.add(Activation('relu'))
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")

    return model


nb_epoch = 5 
model = get_model()
h5_path = '../data/p3_data.h5'
batch_size = 128    # 32 is better, since it will have more weights updates, but 128 is used in this model.
nb_train = HDF5Matrix(h5_path, 'y_train').shape[0]
samples_per_epoch = int(np.floor(nb_train/batch_size) * batch_size)  # make samples_per_epoch in fit_generator fit

# Training model
history = model.fit_generator(batch_generator(h5_path, 'X_train', 'y_train', batch_size),
                              samples_per_epoch=samples_per_epoch,
                              nb_epoch=nb_epoch,
                              callbacks=[EarlyStopping(patience=2)],
                              validation_data=(HDF5Matrix(h5_path, 'X_val'),
                                               HDF5Matrix(h5_path, 'y_val')))

status = 'Closing tensorflow session'
gc.collect()

model.save_weights('./model.h5')
with open('./model.json', 'w') as f:
    json.dump(model.to_json(), f)