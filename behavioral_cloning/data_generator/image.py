#!/usr/bin/python
from __future__ import absolute_import

import cv2
import numpy as np
import threading
import os
import matplotlib.pyplot as plt

DEBUG = True

def _random_flip(x, y):
    if np.random.random() < 0.5:
        x = np.fliplr(x)
        y = -y
    return x, y

def _random_brightness(img):
    image1 = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    random_bright = np.random.uniform(low=0.7, high=1.5)
    image1[:,:,2] = image1[:,:,2] * random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

def _crop_image(x, output_shape):
    return cv2.resize(x[40:140,:,:], (output_shape[1], output_shape[0]))

def _mean_substract(x, rescale):
    return x / rescale - 0.5

def _preprocess(x, rescale, output_shape):
    #print(x.shape)
    x = _crop_image(x, output_shape)
    #x = _mean_substract(x, rescale)
    return x

def _file_to_image(root_path, fname):
    _loc = os.path.join(root_path, fname)
    return cv2.imread(_loc)


class ImageDataGenerator(object):
    """
    ImageDataGenerator is very similar to Keras image preprocessing generator.
    I opted to create my own generator instead because the
    problem requires its unique set of transformations.

    Generate minibatches of training data.
    Acts like a DirectoryIterator
    """
    def __init__(self, target_map, input_shape, output_shape, batch_size, root_path, shuffle=True, training=True):
        self.target_map= target_map
        self.output_shape = output_shape
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.rescale = 1./255
        self.shift_degree = 22.5
        self.lock = threading.Lock()
        self.root_path = root_path
        self.filenames = os.listdir(root_path)
        self.batch_index = 0
        self.N = len(self.target_map.keys())
        self.index_generator = self._flow_index(self.N, batch_size, shuffle)
        self.y_hist = []
        self.training = training

    def reset(self):
        self.batch_index = 0

    # maintain the state
    def _flow_index(self, N, batch_size, shuffle=True):
        while 1:
            if self.batch_index == 0:
                index_array = np.arange(N)
            if shuffle:
                index_array = np.random.permutation(N)

            current_index = (self.batch_index * batch_size) % N
            if N >= current_index + batch_size:
                current_batch_size = batch_size
                self.batch_index += 1
            else:
                current_batch_size = N - current_index
                self.batch_index = 0
            yield (index_array[current_index: current_index + current_batch_size],
               current_index, current_batch_size)

    def __next__(self):
        # To enable multithreading
        with self.lock:
            ix_array, current_index, current_batch_size = next(self.index_generator)
        batch_x = np.zeros([current_batch_size] + list(self.output_shape))
        batch_y = np.zeros(current_batch_size)
        for i, j in enumerate(ix_array):
            fname = self.filenames[j]
            x = self.file_to_image(fname)
            y = self.target_map[fname]

            if self.training:
                # Flip LR + perspective transformation
                x, y = self.transform(x, y)
            # Preprocess
            x = self.preprocess(x)

            batch_x[i,:,:] = x
            batch_y[i] = y
        return batch_x, batch_y

    def __iter__(self):
        return self

    def file_to_image(self, fname):
        return _file_to_image(self.root_path, fname)

    def transform(self, x, y):
        x,y = self.random_flip(x, y)
        x = self.random_brightness(x)
        x = self.random_vertical_shift(x)
        return x, y

    def random_flip(self, x, y):
        return _random_flip(x, y)

    def random_brightness(self, x):
        return _random_brightness(x)

    def random_vertical_shift(self, x):
        h,w,_ = x.shape
        horizon = 2*h/5
        v_shift = np.random.randint(-h//8,h//8)
        # from
        pts1 = np.float32([[0,horizon],[w, horizon],[0,h],[w,h]])
        # to
        pts2 = np.float32([[0,horizon+v_shift],[w,horizon+v_shift],[0,h],[w,h]])
        M = cv2.getPerspectiveTransform(pts1,pts2)
        return cv2.warpPerspective(x, M, (w,h), borderMode=cv2.BORDER_REPLICATE)

    def preprocess(self, x):
        return _preprocess(x, self.rescale, self.output_shape)

    def crop_image(self, x):
        return _crop_image(x, self.output_shape)

    def mean_substract(self, x):
        return _mean_substract(x, self.rescale)
