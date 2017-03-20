from .preprocessing import create_hog_features
import numpy as np
from scipy.ndimage.measurements import label
import cv2
import pickle
import queue
import multiprocessing as mp
from multiprocessing.pool import Pool
from itertools import chain
from matplotlib import pyplot as plt
import uuid

from .preprocessing import *

MODEL_PATH = "saved_models/"
Q_SIZE = 4
DEBUG = True
MULTIPROCESSING = True
COLLECT_DATA = True
Y_CROP_TOP = 375

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
    thresholded = []

    # Crop and Resize
    img = img[y_crop[0]:y_crop[1],:,:]
    _img = cv2.resize(img, (np.int(img.shape[1]/scale), np.int(img.shape[0]/scale)))
    # plt.imshow(_img)
    # plt.show()
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
    image_hist_feature = create_color_hist(_img, path=False).reshape(1,-1)
    # if DEBUG:
    #     print("{} {} Img size {}".format(scale, y_crop, _img.shape))
    #     print("{} {} nb_y_blocks {}, ".format(scale, y_crop, nb_y_blocks))
    #     print("{} {} cells_per_step {}".format(scale, y_crop, cells_per_step))
    #     print("{} {} nb_blocks_per_window {}".format(scale, y_crop, nb_blocks_per_window))
    #     print("{} {} Nb of y steps {}".format(scale, y_crop, nb_y_steps))
    #     print("***")
    for xb in range(nb_x_steps):
        for yb in range(nb_y_steps):
            y = yb * cells_per_step
            x = xb * cells_per_step
            # Extract HOG for this patch
            # Ravel at this step because `feature_vec` is set to false
            hog_features = _hog[:,y : y + nb_blocks_per_window, \
                x : x + nb_blocks_per_window].ravel().reshape(1, -1)

            xleft = x * pix_per_cell
            ytop = y * pix_per_cell

            # Extract the color histogram
            _img_hist = cv2.resize(_img[ytop:ytop+window, xleft:xleft+window], (window,window))
            hist_features = create_color_hist(_img_hist, path=False).reshape(1,-1)
            features = np.concatenate((hog_features, hist_features),axis=1)

            _thresholded = hls_thresholding(_img_hist,2)
            thresholded.append(_thresholded)

            # Scale features and make a prediction
            test_features = scaler.transform(features)
            if svc.predict(test_features) and _thresholded <= 0.9:
                if COLLECT_DATA:
                    # Save randomly to train on false positives later on
                    if np.random.uniform(0,1) < 0.1:
                        img_name = uuid.uuid4().hex
                        # Convert back to RGB
                        img_to_save = cv2.cvtColor(_img_hist, cv2.COLOR_BGR2RGB)
                        cv2.imwrite("images/{}.png".format(img_name),img_to_save)

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
    # Thresdhold summary stats
    # if thresholded:
    #     print("Thresholded saturation summary:")
    #     print(np.histogram(np.array(thresholded),np.linspace(0,1,9))[0])
    return proposed_regions

# channel thresholding
def hls_thresholding(img, channel_ix, threshold=(120,255)):
    """HLS thresholding"""
    # channel in HLS
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    channel = hls[:,:,channel_ix]
    _, binary = cv2.threshold(channel.astype('uint8'), threshold[0], threshold[1], cv2.THRESH_BINARY)
    proportion_of_thresholded = (binary == 0).sum() / (binary.shape[0] * binary.shape[1])
    return proportion_of_thresholded

def non_max_suppression(boxes, threshold=0.3):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []

	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")

	# initialize the list of picked indexes
	pick = []

	# grab the coordinates of the bounding boxes
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]

	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)

	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)

		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])

		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)

		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]

		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > threshold)[0])))

	# return only the bounding boxes that were picked using the
	# integer data type
	return boxes[pick].astype("int")

class Detector(object):
    def __init__(self, orient, image_size, pix_per_cell, cell_per_block):
        self.orient = orient
        self.pix_per_cell = pix_per_cell
        self.cell_per_block = cell_per_block
        self.image = None
        # Derived Fields
        self.heatmap = np.zeros(image_size).astype(np.float)
        self.masked_heatmap = np.zeros(image_size).astype(np.float)
        self.decay = 0.25 # decrement at each step
        self.threshold = 5.
        self.proposed_regions = []
        # Tuple: (pixels, nb_labels found)
        self.labels = None
        self.nb_frames_processed = 0
        # Windowing at various scales
        # Closer, farther
        self.scales = [3, 2, 2, 1]
        self.cells_per_step = [1, 1, 1, 2]
        self.y_crops = [(Y_CROP_TOP,650), (Y_CROP_TOP,600), (Y_CROP_TOP,550), (Y_CROP_TOP,500)]

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
        if MULTIPROCESSING:
            results = []
            for step, scale, y_crop  in zip(self.cells_per_step,
                self.scales, self.y_crops):
                result = self.pool.apply_async(_find_proposed_regions, (
                    self.image, scale, step, y_crop, self.detection_model, self.scaler,
                ))
                results.append(result)
            proposed_regions = list(chain(*[r.get() for r in results]))
        else:
            proposed_regions = []
            for step, scale, y_crop  in zip(self.cells_per_step,
                self.scales, self.y_crops):
                result = _find_proposed_regions(self.image,
                    scale, step, y_crop,
                    self.detection_model, self.scaler)
                proposed_regions.extend(result)
        print("{} proposed regions".format(len(proposed_regions)))
        if proposed_regions:
            self.proposed_regions = proposed_regions

    def update_heatmap(self):
        current_heatmap = np.zeros_like(self.heatmap)
        for box in self.proposed_regions:
            current_heatmap[box[1]:box[3], box[0]:box[2]] += 1.
        print("Current: mean {} max {}".format(current_heatmap.mean(),current_heatmap.max()))
        # heatmap is the memory state for the Detector
        self.heatmap = (1 - self.decay) * self.heatmap + self.decay * current_heatmap
        print("Before: mean {} max {}".format(self.heatmap.mean(),self.heatmap.max()))
        # Thresholding
        self.masked_heatmap = self.heatmap.copy()
        self.masked_heatmap[self.heatmap < self.threshold] = 0
        print("After: mean {} max {}".format(self.masked_heatmap.mean(),self.masked_heatmap.max()))

    def detect(self):
        """
        Threshold the heatmap.
        Reject points that are less than a predetermined threshold
        Find the labels
        """
        self.labels = label(self.masked_heatmap)
        print("Found {} labels".format(self.labels[1]))

    def get_labels(self):
        img = self.image.copy()
        if self.labels:
            for i in range(1, self.labels[1] + 1):
                # Find pixels with each label value
                nonzero = (self.labels[0] == i).nonzero()
                # Define a bounding box based on min/max x and y
                # Plus some padding
                _padding = 10
                _region = ((np.min(nonzero[1]) - _padding,
                            np.min(nonzero[0]) - _padding),
                           (np.max(nonzero[1]) + _padding,
                            np.max(nonzero[0]) + _padding))
                # Draw the box on the image
                cv2.rectangle(img, _region[0], _region[1], (0,0,255), 6)
        # Return the image
        return img

    def show_frame(self):
        plt.figure(figsize=(20,10))
        plt.subplot(4,1,1)
        self.show_proposed_regions()
        plt.subplot(4,1,2)
        self.show_heatmap()
        plt.subplot(4,1,3)
        self.show_masked_heatmap()
        plt.subplot(4,1,4)
        self.show_labeled_image()
        plt.show()


    def show_heatmap(self):
        plt.imshow(self.heatmap,cmap='gray')

    def show_masked_heatmap(self):
        plt.imshow(self.masked_heatmap,cmap='gray')

    def show_labeled_image(self):
        plt.imshow(self.get_labels())

    def show_proposed_regions(self):
        if self.proposed_regions:
            draw_img = self.image.copy()
            for proposed_region in self.proposed_regions:
                x1, y1, x2, y2 = proposed_region
                cv2.rectangle(draw_img,(x1, y1),(x2,y2),(0,0,255),6)
        # Return the image
        plt.imshow(draw_img)

    def process(self, img):
        self.nb_frames_processed += 1
        print(self.nb_frames_processed)
        self.image = img
        self.find_proposed_regions()
        self.update_heatmap()
        self.detect()
        if DEBUG:
            self.show_frame()
        return self.get_labels()
