


from sklearn.svm import LinearSVC
from skimage.feature import hog
import cv2
import numpy as np
import os

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

def read_images():
    vehicle_img_array, non_vehicle_img_array = [], []
    for (path, dirs, files) in os.walk(VEHICLE_IMG_FOLDER):
        _full_path = [path + "/" + _file for _file in files]
        vehicle_img_array.extend(_full_path)
    for (path, dirs, files) in os.walk(NONVEHICLE_IMG_FOLDER):
        _full_path = [path + "/" + _file for _file in files]
        non_vehicle_img_array.extend(_full_path)
    return vehicle_img_array, non_vehicle_img_array

def create_feature_space():
    """
    Read data
    Create the feature space
    """
    vehicle_img_path, non_vehicle_img_path = read_images()
    print("Retrieved the image paths...")
    print("{} images found".format(len(vehicle_img_path) + len(non_vehicle_img_path)))

    # HOG Features
    hog_features_vehicles = np.array([create_hog_features(_img) for _img in vehicle_img_path])
    hog_features_non_vehicles = np.array([create_hog_features(_img) for _img in non_vehicle_img_path])
    print("Created HOG Features...")

    X = np.vstack((hog_features_vehicles, hog_features_non_vehicles)).astype(np.float64)
    y = np.hstack((np.ones(hog_features_vehicles.shape[0]), np.zeros(hog_features_non_vehicles.shape[0])))

    print("Generated Feature Space {}".format(X.shape))
    return X,y
