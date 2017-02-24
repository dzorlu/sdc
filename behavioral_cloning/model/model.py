
from __future__ import absolute_import

from keras.models import Sequential
from keras.regularizers import l2
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, Input, InputLayer, Lambda
from keras import activations, initializations, regularizers, constraints, optimizers

from pathlib import Path
import json
import os


class CNNModel():
    def __init__(self, input_shape, nb_fully_connected, filter_sizes, pool_size, dropout_p, l2_reg):
        self.input_shape = input_shape
        self.filter_sizes = filter_sizes
        self.pool_size = pool_size
        self.dropout_p = dropout_p
        self.n_fc = nb_fully_connected
        self.l2_reg = l2_reg

        """
        Build and return a CNN; details in the comments.
        The intent is a scaled down version of the model from "End to End Learning
        for Self-Driving Cars": https://arxiv.org/abs/1604.07316.
        Args:
        cameraFormat: (3-tuple) Ints to specify the input dimensions (color
            channels, rows, columns).
        Returns:
        A compiled Keras model.
        """
        print("Building model...")
        f1, f2, f3 = self.filter_sizes

        model = Sequential()
        model.add(Lambda(lambda x: x/255.-0.5,input_shape=self.input_shape))
        model.add(Convolution2D(16, f1, f1, input_shape=(32, 128, 3), activation='relu', border_mode='same'))
        model.add(MaxPooling2D(pool_size=self.pool_size))
        model.add(Convolution2D(32, f2, f2, activation='relu', border_mode='same'))
        model.add(MaxPooling2D(pool_size=self.pool_size))
        model.add(Convolution2D(64, f3, f3, activation='relu', border_mode='same'))
        model.add(MaxPooling2D(pool_size=self.pool_size))
        model.add(Convolution2D(64, f3, f3, activation='relu', border_mode='same'))
        model.add(MaxPooling2D(pool_size=self.pool_size))
        model.add(Flatten())
        model.add(Dropout(dropout_p))
        model.add(Dense(self.n_fc, W_regularizer=l2(self.l2_reg), activation='relu'))
        model.add(Dense(self.n_fc//2, W_regularizer=l2(self.l2_reg), activation='relu'))
        model.add(Dense(1))

        model.compile(optimizer=optimizers.Adam(lr=1e-04), loss="mse")

        self.model = model
        print("Compiled the model...")

    def train(self, gen, batch_size, nb_epochs, val_gen):
        # returns a history object
        return self.model.fit_generator(gen,
            samples_per_epoch= gen.batch_size * 50 ,
            nb_epoch = nb_epochs,
            validation_data = val_gen,
            nb_val_samples = batch_size * 5)

    def save(self, file_json, file_weights):
        # saves json and hd5 files onto disk
        if Path(file_json).is_file():
            os.remove(file_json)
        json_string = self.model.to_json()
        with open(file_json,'w' ) as f:
            json.dump(json_string, f)
        if Path(file_weights).is_file():
            os.remove(file_weights)
        self.model.save(file_weights)
