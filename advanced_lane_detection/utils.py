import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from functools import partial

from image_transformation import *

def lane_masking_all(img):
    img = gaussian_blur(img)
    # Color Thresholding -
    # Saturation channel in HLS
    s_binary = hls_thresholding(img, 2)
    # L channel in HLS
    l_binary = hls_thresholding(img, 1)
    # Red
    r_binary = rgb_thresholding(img, 2)
    # Blue
    b_binary = rgb_thresholding(img, 0)
    #Sobel Horizontal and Vertical Gradient with a given kernel size
    x_binary = sobel_thresholding(img, dim = 'x')
    y_binary = sobel_thresholding(img, dim = 'y')
    # Direction Threshold
    d_binary = direction_thresholding(img)
    # Laplacian Threshold
    laplacian_binary = laplacian_thresholding(img)
    # Gray Threshold
    g_binary = gray_thresholding(img)

    #return np.dstack((s_binary, r_binary, x_binary, y_binary, d_binary, l_binary))
    return {'s_binary' : s_binary, 'l_binary':l_binary,
            'b_binary': b_binary, 'g_binary': g_binary,
            'r_binary': r_binary, 'x_binary':x_binary,
            'y_binary' : y_binary, 'd_binary':d_binary,
            'laplacian_binary' : laplacian_binary}

def show_masks(masked):
    plt.figure(figsize=(200,100))
    plt.subplot(3,3,1)
    plt.imshow(masked['s_binary'],cmap='gray')
    plt.title('s_channel',fontsize=FONT_SIZE)

    plt.subplot(3,3,2)
    plt.imshow(masked['l_binary'],cmap='gray')
    plt.title('l_channel',fontsize=FONT_SIZE)

    plt.subplot(3,3,3)
    plt.imshow(masked['r_binary'],cmap='gray')
    plt.title('r_channel',fontsize=FONT_SIZE)

    plt.subplot(3,3,4)
    plt.imshow(masked['x_binary'],cmap='gray')
    plt.title('x_magnitude',fontsize=FONT_SIZE)

    plt.subplot(3,3,5)
    plt.imshow(masked['y_binary'],cmap='gray')
    plt.title('y_magnitude',fontsize=FONT_SIZE)

    plt.subplot(3,3,6)
    plt.imshow(masked['d_binary'],cmap='gray')
    plt.title('direction',fontsize=FONT_SIZE)

    plt.subplot(3,3,7)
    plt.imshow(masked['laplacian_binary'],cmap='gray')
    plt.title('laplacian',fontsize=FONT_SIZE)

    plt.subplot(3,3,8)
    plt.imshow(masked['b_binary'],cmap='gray')
    plt.title('b_channel',fontsize=FONT_SIZE)

    plt.subplot(3,3,9)
    plt.imshow(masked['g_binary'],cmap='gray')
    plt.title('g_binary',fontsize=FONT_SIZE)
