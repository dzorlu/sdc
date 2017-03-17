
from sklearn.svm import LinearSVC
from skimage.feature import hog
import cv2
import numpy as np
import os
import random

ORIENT = 9 # number of bins
PIX_PER_CELL = 8
CELL_PER_BLOCK = 2
HOG_CHANNEL = [0,1,2] # Can be 0, 1, 2

VEHICLE_IMG_FOLDER, NONVEHICLE_IMG_FOLDER = "training_images/vehicles", "training_images/non-vehicles"

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
    if nb_samples:
        random.shuffle(vehicle_img_array)
        random.shuffle(non_vehicle_img_array)
    return vehicle_img_array[:nb_samples//2], non_vehicle_img_array[:nb_samples//2]

def create_feature_space():
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
