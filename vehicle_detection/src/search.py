from .preprocessing import create_hog_features
import numpy as np
from scipy.ndimage.measurements import label
import cv2
import pickle

MODEL_PATH = "saved_models/"

# Define a function you will pass an image
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, color_space='RGB',
                    spatial_size=(32, 32), hist_bins=32,
                    hist_range=(0, 256), orient=9,
                    pix_per_cell=8, cell_per_block=2,
                    hog_channel=0, spatial_feat=True,
                    hist_feat=True, hog_feat=True):

    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        #4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space,
                            spatial_size=spatial_size, hist_bins=hist_bins,
                            orient=orient, pix_per_cell=pix_per_cell,
                            cell_per_block=cell_per_block,
                            hog_channel=hog_channel, spatial_feat=spatial_feat,
                            hist_feat=hist_feat, hog_feat=hog_feat)
        #5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows

def _find_proposed_regions(img, svc, scaler, orient, pix_per_cell, cell_per_block, y_crop=(400,656)):
    """
    Find image regions detected by the trained SVM classifier
    """
    rejection_count, total_count = 0, 0

    # Crop
    img = img[y_crop[0]:y_crop[1],:,:]

    # Define blocks and steps
    nb_x_blocks = (img.shape[1] // pix_per_cell) - 1
    nb_y_blocks = (img.shape[0] // pix_per_cell) - 1

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64 # the size of the sampled image
    nb_blocks_per_window = (window // pix_per_cell) - 1 # Kernel
    #print("nblocks_per_window {}".format(nblocks_per_window))
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nb_x_steps = (nb_x_blocks - nb_blocks_per_window) // cells_per_step
    #print("nb_x_steps {}".format(nb_x_steps))
    nb_y_steps = (nb_y_blocks - nb_blocks_per_window) // cells_per_step
    #print("nb_y_steps {}".format(nb_y_steps))

    # Compute individual channel HOG features for the entire image
    # feature_vec is set to False because we sample from the HOG space
    _hog = create_hog_features(img, path=False, feature_vec=False)
    proposed_regions = []
    for xb in range(nb_x_steps):
        for yb in range(nb_y_steps):
            total_count += 1
            y = yb * cells_per_step
            x = xb * cells_per_step
            # Extract HOG for this patch
            # Ravel at this step because `feature_vec` is set to false
            hog_features = _hog[:,y : y + nb_blocks_per_window, \
                x : x + nb_blocks_per_window].ravel().reshape(1, -1)
            # Scale features and make a prediction
            test_features = scaler.transform(hog_features)
            if svc.predict(test_features):
                xleft = x * pix_per_cell
                ytop = y * pix_per_cell
                xbox_left = np.int(xleft)
                ytop_draw = np.int(ytop)
                win_draw = np.int(window)
                proposed_regions.append(
                    # Tuple denoting the corners
                    (xbox_left,
                    ytop_draw + y_crop[0],
                    xbox_left + win_draw,
                    ytop_draw + win_draw + y_crop[0])
                )
            else:
                rejection_count +=1
    #print("number of regions rejected {} out of {}".format(rejection_count, total_count))
    return proposed_regions

class Detector(object):
    def __init__(self, orient, image_size, pix_per_cell, cell_per_block, y_crop):
        self.orient = orient
        self.pix_per_cell = pix_per_cell
        self.cell_per_block = cell_per_block
        self.y_crop = y_crop
        self.image = None
        # Derived Fields
        self.heatmap = np.zeros(image_size).astype(np.float)
        self.decay = 1. # decrement at each step
        self.threshold = 3.
        self.proposed_regions = None
        # Tuple: (pixels, nb_labels found)
        self.labels = None
        self.nb_frames_processed = 0

        # Load scaler and SVM
        _file = open(MODEL_PATH+"/standard_scaler",'rb')
        # load the object from the file
        self.scaler = pickle.load(_file)
        _file.close()

        _file = open(MODEL_PATH+"/linear_svm",'rb')
        # load the object from the file
        self.detection_model = pickle.load(_file)
        _file.close()

    def find_proposed_regions(self):
        proposed_regions = _find_proposed_regions(self.image,
            self.detection_model, self.scaler, self.orient, self.pix_per_cell,
            self.cell_per_block, self.y_crop)
        print("{} proposed regions".format(len(proposed_regions)))
        self.proposed_regions = proposed_regions

    def update_heat_map(self):
        for box in self.proposed_regions:
            self.heatmap[box[1]:box[3], box[0]:box[2]] += 1.
        # Decaying Factor
        self.heatmap -= self.decay
        self.heatmap[self.heatmap <= self.threshold] = 0

    def detect(self):
        """
        Threshold the heatmap.
        Reject points that are less than a predetermined threshold
        Find the labels
        """
        self.labels = label(self.heatmap)

    def show_labels(self):
        img = self.image.copy()
        if self.labels:
            for i in range(1, self.labels[1] + 1):
                # Find pixels with each label value
                nonzero = (self.labels[0] == i).nonzero()
                # Define a bounding box based on min/max x and y
                _region = ((np.min(nonzero[1]), np.min(nonzero[0])),
                        (np.max(nonzero[1]), np.max(nonzero[0])))
                # Draw the box on the image
                cv2.rectangle(img, _region[0], _region[1], (0,0,255), 6)
        # Return the image
        return img

    def show_proposed_regions(self):
        if self.proposed_regions:
            draw_img = self.image.copy()
            for proposed_region in self.proposed_regions:
                # (0, 416, 64, 480)
                x1, y1, x2, y2 = proposed_region
                cv2.rectangle(draw_img,(x1, y1),(x2,y2),(0,0,255),6)
        # Return the image
        return draw_img

    def process(self, img):
        self.image = img
        self.nb_frames_processed += 1
        self.find_proposed_regions()
        self.update_heat_map()
        self.detect()
        return self.show_labels()
