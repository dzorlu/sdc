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
import tensorflow as tf

from .preprocessing import *

MODEL_PATH = "saved_models/"
CHECK_POINT_NAME = "model.ckpt"
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
        svc,
        saver,
        scaler,
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

            # Scale features and make a prediction
            test_features = scaler.transform(features)
            if svc.predict(test_features) and _thresholded <= 0.75:

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

                _padding = 25
                proposed_regions.append({
                    'proposed_image': _img_hist,
                    # Tuple denoting the corners
                    'corners':
                        (xbox_left,
                        ytop_draw + y_crop[0],
                        xbox_left + win_draw + _padding,
                        ytop_draw + win_draw + y_crop[0] + _padding)
                }
                )
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

class Detector(object):
    def __init__(self, orient, image_size, pix_per_cell, cell_per_block):
        self.orient = orient
        self.pix_per_cell = pix_per_cell
        self.cell_per_block = cell_per_block
        self.image = None
        # Derived Fields
        self.heatmap = np.zeros(image_size).astype(np.float)
        self.masked_heatmap = np.zeros(image_size).astype(np.float)
        self.decay = 0.2 # decrement at each step
        self.threshold = 1.2
        self.power = 2 # the weight placed on new observations with overlapping windows
        self.proposed_regions = []
        self.accepted_regions = []
        # Tuple: (pixels, nb_labels found)
        self.labels = None
        self.nb_frames_processed = 0
        # Windowing at various scales
        # Closer, farther
        self.scales = [3, 2, 2, 1]
        self.cells_per_step = [1, 1, 1, 1]
        self.y_crops = [(Y_CROP_TOP,650), (Y_CROP_TOP,600), (Y_CROP_TOP,550), (Y_CROP_TOP,500)]

        # Load scaler and SVM
        _file = open(MODEL_PATH+"/standard_scaler",'rb')
        # load the object from the file
        self.scaler = pickle.load(_file)
        _file.close()

        # load the models
        _file = open(MODEL_PATH+"/linear_svm",'rb')
        self.detection_model = pickle.load(_file)
        _file.close()

        # Retrieve the graph
        self.saver = self.recover_model(MODEL_PATH + CHECK_POINT_NAME + ".meta")
        # Multi-processing
        self.pool = mp.Pool(processes=Q_SIZE)

    def recover_model(self, meta_filename):
        return tf.train.import_meta_graph(meta_filename, clear_devices=True)

    def predict(self, _X):
        """Predict whether the image contains a vehicle or not"""
        def mean_substract(img):
            return (img / 255.).astype(np.float32)
        _X = mean_substract(_X)
        with tf.Session() as sess:
            # Restore the variable within a session
            latest_checkpoint = tf.train.latest_checkpoint(MODEL_PATH)
            self.saver.restore(sess, latest_checkpoint)
            logits_ops = tf.get_collection("logits")[0]
            input_tensor = tf.get_collection("input")[0]
            _predictions = sess.run(logits_ops, feed_dict={input_tensor: _X})
        _predictions = np.array(_predictions)
        return _predictions.reshape(-1,2).argmax(axis=1)

    def find_proposed_regions(self):
        if MULTIPROCESSING:
            results = []
            for step, scale, y_crop  in zip(self.cells_per_step,
                self.scales, self.y_crops):
                result = self.pool.apply_async(_find_proposed_regions, (
                    self.image, scale, step, y_crop, self.detection_model, self.saver, self.scaler,
                ))
                results.append(result)
            proposed_regions = list(chain(*[r.get() for r in results]))
        else:
            proposed_regions = []
            for step, scale, y_crop  in zip(self.cells_per_step,
                self.scales, self.y_crops):
                result = _find_proposed_regions(self.image,
                    scale, step, y_crop,
                    self.detection_model, self.saver, self.scaler)
                proposed_regions.extend(result)
        print("{} proposed regions".format(len(proposed_regions)))

        # Run the second classifier on proposed images
        # 1 == vehicle, 0 == non-vehicle
        self.proposed_regions = [proposal['corners'] for proposal in proposed_regions]
        _images = np.array([pr['proposed_image'] for pr in proposed_regions])
        self.accepted_regions = []
        if len(_images) > 0:
            _labels = self.predict(_images)
            filtered_regions = [_proposed_region['corners'] for _label, _proposed_region in zip(_labels, proposed_regions) if _label]
            if filtered_regions:
                print("{} accepted regions".format(len(filtered_regions)))
                self.accepted_regions = filtered_regions

    def update_heatmap(self):
        """
        Update the heatmap. More confidence in multiple detections
        Threshold the heatmap.
        Reject points that are less than a predetermined threshold
        """
        current_heatmap = np.zeros_like(self.heatmap)
        for box in self.accepted_regions:
            current_heatmap[box[1]:box[3], box[0]:box[2]] += 1.
        # Boost stops with multiple triangles.
        current_heatmap = current_heatmap ** self.power
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
        Find the labels
        If no detection, keep the labels
        """
        _labels = label(self.masked_heatmap)
        print("Found {} label(s)".format(_labels[1]))
        self.labels = _labels

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
        plt.subplot(5,1,1)
        self.show_proposed_regions()
        plt.subplot(5,1,2)
        self.show_accepted_regions()
        plt.subplot(5,1,3)
        self.show_heatmap()
        plt.subplot(5,1,4)
        self.show_masked_heatmap()
        plt.subplot(5,1,5)
        self.show_labeled_image()
        plt.show()


    def show_heatmap(self):
        plt.imshow(self.heatmap,cmap='gray')

    def show_masked_heatmap(self):
        plt.imshow(self.masked_heatmap,cmap='gray')

    def show_labeled_image(self):
        plt.imshow(self.get_labels())

    def show_proposed_regions(self):
        draw_img = self.image.copy()
        for proposed_region in self.proposed_regions:
            x1, y1, x2, y2 = proposed_region
            cv2.rectangle(draw_img,(x1, y1),(x2,y2),(0,0,255),6)
        # Return the image
        plt.imshow(draw_img)

    def show_accepted_regions(self):
        draw_img = self.image.copy()
        for accepted_region in self.accepted_regions:
            x1, y1, x2, y2 = accepted_region
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
