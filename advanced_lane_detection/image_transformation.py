import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

"""Transformation Pipeline"""


calibration_image_src = 'camera_cal/calibration*.jpg'
FONT_SIZE = 200

def calibrate_camera(calibration_image_src = calibration_image_src, ):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob(calibration_image_src)

    # Step through the list and search for chessboard corners
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp) # image
            imgpoints.append(corners) #real world 2D

    return {'objpoints': objpoints, 'imgpoints': imgpoints}

def undistort_image(img, pts):
    objpoints, imgpoints =  pts['objpoints'], pts['imgpoints']
    _shape = img.shape if len(img.shape) == 2 else img.shape[::2]
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, _shape,None,None)
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    return dst

def gaussian_blur(img, kernel_size=7):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def sobel_thresholding(img, kernel_size=5, threshold=(30,255), dim='x'):
    """one dimensional thresholding"""
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    x, y = (1, 0) if dim is 'x' else (0,1)
    sobel = cv2.Sobel(gray, cv2.CV_64F, x, y, ksize = kernel_size)
    sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255*sobel/np.max(sobel))
    _, binary = cv2.threshold(scaled_sobel.astype('uint8'), threshold[0], threshold[1], cv2.THRESH_BINARY)
    return binary

def direction_thresholding(img, kernel_size=15, threshold = (0.9, 1.1)):
    """threshold by angle of the gradient"""
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel_size)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel_size)
    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    _, binary = cv2.threshold(absgraddir.astype('uint8'), threshold[0], threshold[1], cv2.THRESH_BINARY)
    return binary

# color channel thresholding
def hls_thresholding(img, channel_ix, threshold=(150,255)):
    """HLS thresholding"""
    # channel in HLS
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    channel = hls[:,:,channel_ix]
    _, binary = cv2.threshold(channel.astype('uint8'), threshold[0], threshold[1], cv2.THRESH_BINARY)
    return binary

# color channel thresholding
def rgb_thresholding(img, channel_ix, threshold=(170,255)):
    """R thresholding"""
    # R channel in BGR = cv2.COLOR_BGR2GRAY
    channel = img[:,:,channel_ix]
    _, binary = cv2.threshold(channel.astype('uint8'), threshold[0], threshold[1], cv2.THRESH_BINARY)
    return binary

# # laplacian threshold
def laplacian_thresholding(img, kernel=15):
    """Laplacian thresholding"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray,cv2.CV_32F,ksize= kernel)
    return (laplacian < 0.15 * np.min(laplacian)).astype(np.uint8)

# gray channel threshold
def gray_thresholding(img, threshold=(130,255)):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, gray_binary = cv2.threshold(gray.astype('uint8'), threshold[0], threshold[1], cv2.THRESH_BINARY)
    return gray_binary

def define_vertices(img):
    imshape = img.shape
    vertices = np.array([[(0,imshape[0]), (imshape[1]/2., imshape[0]/2.), (imshape[1],imshape[0])]], dtype=np.int32)
    if vertices.shape[1]:
        vertices = [vertices]
    return vertices

def region_of_interest(img):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    vertices = define_vertices(img)
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    # additional layer of cropping. Take 70% and lower on the y-axis
    crop = range(0, int(7 * img.shape[0] / 10),1)
    masked_image[crop,:] = 0
    return masked_image

def lane_masking(img):
    img = gaussian_blur(img)

    s_binary = hls_thresholding(img, 2)
    gray_binary = gray_thresholding(img)
    laplacian_binary = laplacian_thresholding(img)
    # AND following OR gate
    combined_binary = cv2.bitwise_and(laplacian_binary, cv2.bitwise_or(s_binary, gray_binary))
    return combined_binary

def fit_lanes(masked_image):
    # determine the mid point along x-axis
    image_shape = masked_image.shape
    mid_point = image_shape[1]/2
    # index
    ix = masked_image.nonzero()

    left_xs = np.arange(0, mid_point, 1).reshape(-1,1)
    right_xs = np.arange(mid_point, image_shape[1], 1).reshape(-1,1)

    points = [(x,y) for y,x in zip(ix[0],ix[1])]
    # linear regression for left and right space
    left_points = np.array(list(filter(lambda x: x[0] < mid_point, points )))
    right_points = np.array(list(filter(lambda x: x[0] >= mid_point, points )))
    lr_left, lr_right = LinearRegression(), LinearRegression()
    lr_right.fit(right_points[:,0].reshape(-1,1), right_points[:,1].reshape(-1,1))
    lr_left.fit(left_points[:,0].reshape(-1,1), left_points[:,1].reshape(-1,1))
    # prediction for left and right space
    left_ys = lr_left.predict(left_xs).reshape(-1,)
    right_ys = lr_right.predict(right_xs).reshape(-1,)
    left_xs = left_xs.reshape(-1,)
    right_xs = right_xs.reshape(-1,)
    # Mask Y values
    points_left = np.array(list(filter(lambda p: p[1] > image_shape[0]//2 and p[1] < image_shape[0] , zip(left_xs,left_ys))))
    points_right = np.array(list(filter(lambda p: p[1] > image_shape[0]//2 and p[1] < image_shape[0], zip(right_xs,right_ys))))

    return points_left, points_right

def retrieve_src_points(left, right, shape):
    y_cutoff = 7 * shape // 10
    left_cutoff_ix = (left[:,1] > y_cutoff).nonzero()[0].max()
    right_cutoff_ix = (right[:,1] > y_cutoff).nonzero()[0].min()
    p1, p2 = left[left_cutoff_ix,], right[right_cutoff_ix,]

    # Retreieve the trapezoid for perspective transformation
    # We can use the points for all images
    l = {}
    l1, l2 = np.array([int(left[:,0].min()), int(left[:,1].max())]), p1

    r = {}
    r1, r2 = np.array([int(right[:,0].max()), int(right[:,1].max())]), p2

    return np.float32([l1, l2, r1, r2])

def histogram_filter(img, offset = 50):
    filtered = img.copy()
    _hist = filtered.sum(axis=0)
    middlepoint = filtered.shape[1] // 2
    left_max_ix, right_max_ix = _hist[:middlepoint].argmax(), _hist[middlepoint:].argmax() + middlepoint
    left_range, right_range = (left_max_ix - offset, left_max_ix + offset), (right_max_ix - offset, right_max_ix + offset)
    filtered[:,:left_range[0]] = 0
    filtered[:,left_range[1]:right_range[0]] = 0
    filtered[:,right_range[1]:] = 0
    return filtered

def setup_transformation_pipeline(offset=10):
    """
    Set up the transformation pipeline
    Encapsulate the camera distortion and
    transformation pipeline that includes warping of the detected lane points
    """
    pts = calibrate_camera()
    images = glob.glob("test_images/*")
    # Pick the image with a straight lane to calibrate the camera
    img  = cv2.imread(images[0])
    # run the same pipeline
    dst = undistort_image(img, pts)
    masked_img = lane_masking(dst)
    _img = region_of_interest(masked_img)
    # instead of polynomial fit
    # use linear regression to determine the src for perspective transformation
    left,right = fit_lanes(_img)
    src = retrieve_src_points(left, right, masked_img.shape[0])
    dst = np.float32([src[0], (src[0][0], offset), src[2], (src[2][0], offset)])
    return TransformationPipeline(pts, src, dst)

class PerspectiveTransformer:
    def __init__(self, src, dst):
        self.src = src #both src and dst should be mappings that are representative of a straight lane
        self.dst = dst
        self.M = cv2.getPerspectiveTransform(src, dst)
        self.M_inv = cv2.getPerspectiveTransform(dst, src)

    def transform(self, img):
        return cv2.warpPerspective(img, self.M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)

    def inverse_transform(self, img):
        return cv2.warpPerspective(img, self.M_inv, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)


class TransformationPipeline():
    def __init__(self, camera_calibration, src, dst ):
        self.camera_calibration = camera_calibration
        self.perspective_transformer = PerspectiveTransformer(src, dst)

    def transform(self, img):
        _img = self.undistort_image(img)
        binary_img = self.lane_masking(_img)
        binary_img_filtered = self.region_of_interest(binary_img)
        warped_image = self.perspective_transform(binary_img_filtered)
        filtered_warped_image = self.histogram_filter(warped_image)
        return filtered_warped_image

    def undistort_image(self, img):
        return undistort_image(img, self.camera_calibration)

    def lane_masking(self, img):
        return lane_masking(img)

    def region_of_interest(self, img):
        # Filters the image for the lower trapezoid
        return region_of_interest(img)

    def perspective_transform(self, img):
        return self.perspective_transformer.transform(img)

    def inverse_perspective_transform(self, img):
        return self.perspective_transformer.inverse_transform(img)

    def histogram_filter(self, img):
        return histogram_filter(img)
