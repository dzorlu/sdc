
from sklearn.svm import LinearSVC
from skimage.feature import hog
import cv2
import numpy as np
import os
import random
from shutil import copy2

import uuid

ORIENT = 9 # number of bins
PIX_PER_CELL = 8
CELL_PER_BLOCK = 2
HOG_CHANNEL = [0,1,2] # Can be 0, 1, 2

VEHICLE_IMG_FOLDER, NONVEHICLE_IMG_FOLDER = "training_images/vehicles", "training_images/non-vehicles"
FALSE_POSITIVES = "training_images/false-positives"
CNN_VEHICLE_FOLDER = "training_images/cnn/vehicles/"
CNN_NON_VEHICLE_FOLDER = "training_images/cnn/non-vehicles/"

# Define a function to return HOG features and visualization
def create_hog_features(img_path,
                        path = True,
                        orient=ORIENT,
                        pix_per_cell=PIX_PER_CELL,
                        cell_per_block=CELL_PER_BLOCK,
                        hog_channel = HOG_CHANNEL,
                        vis=False,
                        feature_vec=True):
    """
    """
    # Achtung -> imread reads as BGR
    if path:
        img = cv2.imread(img_path)
    else:
        img = img_path

    if vis == True:
        # Pick a single dim
        features, hog_image = hog(img[:,:,1], orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False,
                                  visualise=True, feature_vector=False)
        return features, hog_image
    else:
        hog_features = []

        for i, channel in enumerate(hog_channel):
            features = hog(img[:,:,channel], orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                           cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False,
                           visualise=False, feature_vector = feature_vec)
            hog_features.append(features)
        # Coerce the List of vectors
        if feature_vec:
            return np.ravel(hog_features)
        else:
            return np.array(hog_features)

def create_color_hist(img_path, nbins=32, bins_range=(0, 256),path=True):
    # Achtung -> imread reads as BGR
    if path:
        img = cv2.imread(img_path)
    else:
        img = img_path

    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

def read_images(nb_samples = 10000):
    vehicle_img_array, non_vehicle_img_array = [], []
    for (path, dirs, files) in os.walk(VEHICLE_IMG_FOLDER):
        _full_path = [path + "/" + _file for _file in files]
        vehicle_img_array.extend(_full_path)
    for (path, dirs, files) in os.walk(NONVEHICLE_IMG_FOLDER):
        _full_path = [path + "/" + _file for _file in files]
        non_vehicle_img_array.extend(_full_path)

    _false_positives = [FALSE_POSITIVES + "/" + _file for _file in os.listdir(FALSE_POSITIVES)]
    print("{} false positives found".format(len(_false_positives)))

    if nb_samples:
        random.shuffle(vehicle_img_array)
        random.shuffle(non_vehicle_img_array)
    # Always pick false positives for training
    non_vehicle_img_array = _false_positives + non_vehicle_img_array
    return vehicle_img_array[:nb_samples//2], non_vehicle_img_array[:nb_samples//2]

def create_feature_space_for_fast_classifier():
    """
    Read data
    Create the feature space
    """
    vehicle_img_path, non_vehicle_img_path = read_images(10000)
    print("Retrieved the image paths...")
    print("{} images found".format(len(vehicle_img_path) + len(non_vehicle_img_path)))

    ## HOG Features
    hog_features_vehicles = np.array([create_hog_features(_img) for _img in vehicle_img_path])
    hog_features_non_vehicles = np.array([create_hog_features(_img) for _img in non_vehicle_img_path])
    print("Created HOG Features...")

    ## Created Color Histograms
    color_vehicles = np.array([create_color_hist(_img) for _img in vehicle_img_path])
    color_non_vehicles = np.array([create_color_hist(_img) for _img in non_vehicle_img_path])
    print("Created Color Histogram Features...")

    vehicles = np.concatenate((hog_features_vehicles, color_vehicles),axis=1)
    del hog_features_vehicles, color_vehicles
    non_vehicles = np.concatenate((hog_features_non_vehicles, color_non_vehicles),axis=1)
    del hog_features_non_vehicles, color_non_vehicles

    X = np.vstack((vehicles, non_vehicles)).astype(np.float64)
    y = np.hstack((np.ones(vehicles.shape[0]), np.zeros(non_vehicles.shape[0])))

    print("Generated Feature Space {}".format(X.shape))
    return X,y

def create_feature_space_for_cnn():
    """
    Read data
    Create the feature space
    """
    vehicle_img_path, non_vehicle_img_path = read_images(10000)
    print("Retrieved the image paths...")
    print("{} images found".format(len(vehicle_img_path) + len(non_vehicle_img_path)))

    ## Features
    vehicles = np.array([cv2.imread(_img) for _img in vehicle_img_path])
    non_vehicles = np.array([cv2.imread(_img) for _img in non_vehicle_img_path])
    print("Loaded images...")

    X = np.vstack((vehicles, non_vehicles)).astype(np.float64)
    y = np.hstack((np.ones(vehicles.shape[0]), np.zeros(non_vehicles.shape[0])))

    print("Generated Feature Space {}".format(X.shape))
    return X,y

def copy_images():
    vehicle_img_array, non_vehicle_img_array = [], []
    for (path, dirs, files) in os.walk(VEHICLE_IMG_FOLDER):
        _full_path = [path + "/" + _file for _file in files]
        vehicle_img_array.extend(_full_path)
    for (path, dirs, files) in os.walk(NONVEHICLE_IMG_FOLDER):
        _full_path = [path + "/" + _file for _file in files]
        non_vehicle_img_array.extend(_full_path)

    print("{} images found".format(len(vehicle_img_array) + len(non_vehicle_img_array)))

    _false_positives = [FALSE_POSITIVES + "/" + _file for _file in os.listdir(FALSE_POSITIVES)]

    for _img in vehicle_img_array:
        dst = CNN_VEHICLE_FOLDER + uuid.uuid4().hex + ".png"
        copy2(_img, dst)

    for _img in non_vehicle_img_array:
        dst = CNN_NON_VEHICLE_FOLDER + uuid.uuid4().hex + ".png"
        copy2(_img, dst)

    for _img in _false_positives:
        dst = CNN_NON_VEHICLE_FOLDER + uuid.uuid4().hex + ".png"
        copy2(_img, dst)

    print("Thanks for playing")

class ImageGenerator():
    def __init__(self, batch_size, x, y):
        self.x = x
        self.y = y
        self.batch_index = 0
        self.batch_size = batch_size
        self.shift_degree = 22.5
        self.N = len(x)
        self.shape = (32, 32)
        self.index_generator = self._flow_index(self.N, batch_size)

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

    def __iter__(self):
        return self

    def __next__(self):
        ix_array, current_index, current_batch_size = next(self.index_generator)
        batch_x = np.zeros(tuple([current_batch_size] + list(self.x.shape)[1:]))
        for i, j in enumerate(ix_array):
            x = self.x[j]
            x = self.mean_substract(x)
            batch_x[i] = x
        batch_y = self.y[ix_array]
        return batch_x, batch_y

    def mean_substract(self,img):
        #return img - np.mean(img)
        #Scale the features to be [0,1]
        return (img / 255.).astype(np.float32)
