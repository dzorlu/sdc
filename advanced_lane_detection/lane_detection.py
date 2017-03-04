import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from functools import partial

from image_transformation import *
from kalman_filters import KalmanFilter1D

FONT = cv2.FONT_HERSHEY_SIMPLEX

def identify_points(masked_image):
    """Retrieve the index points for left and right lanes"""
    image_shape = masked_image.shape
    mid_point = image_shape[1] // 2
    ix = masked_image.nonzero()
    points = [(x,y) for y,x in zip(ix[0],ix[1])]
    left_points = np.array(list(filter(lambda x: x[0] < mid_point, points )))
    right_points = np.array(list(filter(lambda x: x[0] >= mid_point, points )))
    return left_points, right_points


def draw_lanes(image, lines, color= [255, 0, 0], thickness=10):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))
    return newwarp


class Line(KalmanFilter1D):
    """
    Line is a state that keeps track of lane elements using 1D Kalman Filters
    Computes the radius of the curvature after fitting polynomial lanes.

    Reject outliers.

    """
    def __init__(self, initial_state = 1280 // 4, process_noise = 1, measurement_noise = 10):
        # was the line detected in the last iteration?
        self.detected = False
        # fitted x values of the last n fits of the line
        self.x = []
        # y. does not change
        self.y = np.linspace(0,720,721)
        #average x values of the fitted line over the last n iterations
        self.wx = None
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #radius of curvature of the line in pixel units
        self.radius_of_curvature = None
        #the distance of the line to corner of the image
        self.base_position = None
        # detected line pixels coordinates
        # this is the raw data
        self.pts = []
        # For real world transformation
        # meters per pixel in y dimension
        self.ym_per_pix = 30/720
        # 700 because it is the distance between lanes
        # meters per pixel in x dimension.
        self.xm_per_pix = 3.7/700
        self.detection_count = (0,0) # True, False
        self.rejected_images = 0
        ##Kalman Filter
        self.kalman_state = KalmanFilter1D(initial_state, linear_state, process_noise)
        super(Line, self).__init__(**kwargs)


    def set_curve_radius(self, order=2):
        """Curve Radius Based on Smoothed Lane Lines"""
        fit_cr = np.polyfit(self.y * self.ym_per_pix, self.wx * self.xm_per_pix, order)
        y_max = np.max(self.y)
        curverad = ((1 + (2*fit_cr[0]*y_max + fit_cr[1])**2)**1.5) / abs(2*fit_cr[0])
        self.radius_of_curvature = curverad

    def fit_poly_lanes(self, order = 2, pixel_curvature = True):
        """Fit a polynomial based on current points"""
        # poly fit fit(y,x) - reversed for this particular problem
        _fit = np.polyfit(self.pts[:,1], self.pts[:,0],order)
        self.current_fit = _fit

    def set_fitted_x(self):
        a,b,c = self.current_fit
        self.x = a * self.y**2 + b * self.y+ c

    def update(self):
        self.x
        # Initial state
        # if self.detection_count[0]<=1:
        #     self.wx = self.x
        # else:
        #     self.wx = w * self.wx + (1-w) * self.x

    def set_base_position(self):
        """ Base position with respect to zero value on x-axis"""
        y_max = np.max(self.y)
        a,b,c = self.current_fit
        self.base_position = a * y_max**2 + b * y_max + c

    def increment_detection_count(self):
        """Increment the detection count - times input was available"""
        _cnt = self.detection_count
        if len(self.pts) >0:
            # Success
            self.detection_count = (_cnt[0]+1, _cnt[1])
            self.detected = True
        else:
            # Failure
            self.detection_count = (_cnt[0], _cnt[1]+1)
            self.detected = False

    def process_image(self, pts):
        self.pts = pts
        self.increment_detection_count()
        self.fit_poly_lanes()
        self.set_fitted_x()
        # kalman update baseline next step
        self.update()
        # Using updated step, calculate the following:
        # curvature update
        self.set_curve_radius()
        # base position update
        self.set_base_position()


def overlay_detected_lane(img, transformer, warped, left, right, show_weighted = True):
    # TODO: Apply it on undiscorted image
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    if show_weighted:
        pts_left = np.array([np.transpose(np.vstack([left.wx, left.y]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right.wx, right.y])))])
    else:
        pts_left = np.array([np.transpose(np.vstack([left.x, left.y]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right.x, right.y])))])
    pts = np.hstack((pts_left, pts_right))
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    # Inverse Transformation
    M_inv = transformer.perspective_transformer.M_inv
    new_warp = cv2.warpPerspective(color_warp, M_inv, (warped.shape[1], warped.shape[0]))
    # distort back
    new_img = cv2.addWeighted(img, 1, new_warp, 0.4, 0)
    # annotate curvature
    mean_curvature = np.mean([left.radius_of_curvature, right.radius_of_curvature])
    _str = "radius of curvature: {} km".format(round(mean_curvature / 1e3,2))
    cv2.putText(new_img, _str, (10,40), FONT, 1,(255,255,255), 2, cv2.LINE_AA)
    # vehicle distance from the center of the image
    _mid = abs((img.shape[1] / 2) - ((left.base_position + right.base_position) / 2)) * left.xm_per_pix
    _str = "distance from center: {} m".format(round(_mid, 3))
    cv2.putText(new_img, _str, (10,70), FONT, 1,(255,255,255), 2, cv2.LINE_AA)
    # detection
    _str = "lane detected? Left: {} Right: {}".format(left.detected, right.detected)
    cv2.putText(new_img, _str, (10,100), FONT, 1,(255,255,255), 2, cv2.LINE_AA)
    return new_img

def process(img, transformer, left, right):
    warped_image = transformer.transform(img)
    # separete points into left and right
    left_points, right_points = identify_points(warped_image)
    # update step. process each half seperately.
    left.process_image(left_points)
    right.process_image(right_points)
    # draw - recast the x and y points into usable format for cv2.fillPoly()
    new_img = overlay_detected_lane(img, transformer, warped_image, left, right)
    return new_img

if __name__ == '__main__':

    in_file = arguments['<input_video>']
    out_file = arguments['<output_video>']
    left, right = Line(), Line()

    print("Prepare the transformation pipeline for image preprocessing")
    transformer = setup_transformation_pipeline()
    fun = lambda x: process(x, transformer, left, right)

    print('Processing video ...')
    clip2 = VideoFileClip(in_file)

    vid_clip = clip2.fl_image(warp)
    vid_clip.write_videofile(out_file, audio=False)
