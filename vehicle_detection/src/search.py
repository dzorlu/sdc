from .preprocessing import create_hog_features
import numpy as np
from scipy.ndimage.measurements import label
import cv2
import pickle
import queue
import multiprocessing as mp
from multiprocessing.pool import Pool
from itertools import chain

MODEL_PATH = "saved_models/"
Q_SIZE = 3

# Define a function you will pass an image
# and the list of windows to be searched (output of slide_windows())
def _find_proposed_regions(img,
        scale,
        cells_per_step ,
        y_crop,
        svc, scaler,
        orient = 9,
        pix_per_cell = 8,
        cell_per_block = 2,
        window = 64,
        ):
    """
    Find image regions detected by the trained SVM classifier
    pix_per_cell = 8
    orient = 9
    cell_per_block = 2
    """

    # Crop and Resize
    img = img[y_crop[0]:y_crop[1],:,:]
    _img = cv2.resize(img, (np.int(img.shape[1]/scale), np.int(img.shape[0]/scale)))
    # Define blocks and steps
    nb_x_blocks = (_img.shape[1] // pix_per_cell) - 1
    nb_y_blocks = (_img.shape[0] // pix_per_cell) - 1

    nb_blocks_per_window = (window // pix_per_cell) - 1 # Kernel
    #print("nblocks_per_window {}".format(nblocks_per_window))
    nb_x_steps = (nb_x_blocks - nb_blocks_per_window) // cells_per_step
    #print("nb_x_steps {}".format(nb_x_steps))
    nb_y_steps = (nb_y_blocks - nb_blocks_per_window) // cells_per_step
    #print("nb_y_steps {}".format(nb_y_steps))

    # Compute individual channel HOG features for the entire image
    # feature_vec is set to False because we sample from the HOG space
    _hog = create_hog_features(_img, path=False, feature_vec=False)
    proposed_regions = []
    for xb in range(nb_x_steps):
        for yb in range(nb_y_steps):
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

                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)

                proposed_regions.append(
                    # Tuple denoting the corners
                    (xbox_left,
                    ytop_draw + y_crop[0],
                    xbox_left + win_draw,
                    ytop_draw + win_draw + y_crop[0])
                )
    return proposed_regions

class Detector(object):
    def __init__(self, orient, image_size, pix_per_cell, cell_per_block):
        self.orient = orient
        self.pix_per_cell = pix_per_cell
        self.cell_per_block = cell_per_block
        self.image = None
        # Derived Fields
        self.heatmap = np.zeros(image_size).astype(np.float)
        self.decay = 0.5 # decrement at each step
        self.threshold = 20.
        self.proposed_regions = []
        # Tuple: (pixels, nb_labels found)
        self.labels = None
        self.nb_frames_processed = 0
        # Windowing at various scales
        # Closer, farther
        self.scales = [2.5, 1.5, 1.25]
        self.cells_per_step = [2, 1, 1]
        self.y_crops = [(400,656), (400,550), (400,500)]

        # Load scaler and SVM
        _file = open(MODEL_PATH+"/standard_scaler",'rb')
        # load the object from the file
        self.scaler = pickle.load(_file)
        _file.close()

        _file = open(MODEL_PATH+"/linear_svm",'rb')
        # load the object from the file
        self.detection_model = pickle.load(_file)
        _file.close()

        self.pool = mp.Pool(processes=Q_SIZE)


    def find_proposed_regions(self):

        results = []
        for step, scale, y_crop  in zip(self.cells_per_step,
            self.scales, self.y_crops):
            result = self.pool.apply_async(_find_proposed_regions, (
                self.image, scale, step, y_crop, self.detection_model, self.scaler,
            ))
            results.append(result)

        proposed_regions = list(chain(*[r.get() for r in results]))
        print("{} proposed regions".format(len(proposed_regions)))
        if proposed_regions:
            self.proposed_regions.extend(proposed_regions)

    def update_heatmap(self):
        current_heatmap = np.zeros_like(self.heatmap)
        for box in self.proposed_regions:
            current_heatmap[box[1]:box[3], box[0]:box[2]] += 1.
        # Decaying Factor
        if self.nb_frames_processed > 25:
            self.heatmap = self.decay * self.heatmap + (1 - self.decay) * current_heatmap
        else:
            self.heatmap = current_heatmap
        # Thresholding
        self.heatmap[self.heatmap <= self.threshold] = 0
        print("Average heatmap value: ".format(self.heatmap.mean()))

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
        self.nb_frames_processed += 1
        print(self.nb_frames_processed)
        self.image = img
        self.find_proposed_regions()
        self.update_heatmap()
        self.detect()
        return self.show_labels()
