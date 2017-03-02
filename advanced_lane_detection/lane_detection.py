import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from functools import partial

from image_transformation import *

FONT = cv2.FONT_HERSHEY_SIMPLEX

def detect_points(masked_image):
    """Retrieve the index points for left and right lanes"""
    image_shape = masked_image.shape
    mid_point = image_shape[1] // 2
    ix = masked_image.nonzero()
    points = [(x,y) for y,x in zip(ix[0],ix[1])]
    left_points = np.array(list(filter(lambda x: x[0] < mid_point, points )))
    right_points = np.array(list(filter(lambda x: x[0] >= mid_point, points )))
    return left_points, right_points


def measure_curvature(_fit, ymax=720):
    curverad = ((1 + (2*_fit[0]*ymax + _fit[1])**2)**1.5) / np.absolute(2*_fit[0])
    return curverad

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


class Line(object):
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # fitted x values of the last n fits of the line
        self.x = []
        # y. does not change
        self.y = np.linspace(0,720,721)
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #the distance of the line to corner of the image
        self.base_position = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        # detected line pixels coordinates
        # this is the raw data
        self.pts = None
        # For real world transformation
        # meters per pixel in y dimension
        self.ym_per_pix = 30/720
        # 700 because it is the distance between lanes
        # meters per pixel in x dimension.
        self.xm_per_pix = 3.7/700

    def set_curve_radius(self, order=2):
        fit_cr = np.polyfit(self.y * self.ym_per_pix, self.x * self.xm_per_pix, order)
        y_max = np.max(self.y)
        curverad = ((1 + (2*fit_cr[0]*y_max + fit_cr[1])**2)**1.5) / abs(2*fit_cr[0])
        self.radius_of_curvature = curverad

    def fit_poly_lanes(self, order = 2, pixel_curvature = True):
        # poly fit fit(y,x) - reversed for this particular problem
        _fit = np.polyfit(self.pts[:,1], self.pts[:,0],order)
        self.current_fit = _fit

    def set_fitted_x(self):
        a,b,c = self.current_fit
        self.x = a * self.y**2 + b * self.y+ c

    def set_base_position(self):
        """ Base position with respect to zero value on x-axis"""
        y_max = np.max(self.y)
        a,b,c = self.current_fit
        self.base_position = a * y_max**2 + b * y_max + c

    def process_image(self, pts):
        self.pts = pts
        self.fit_poly_lanes()
        self.set_fitted_x()
        self.set_curve_radius()
        self.set_base_position()

def overlay_detected_lane(img, transformer, warped, left, right):
    # TODO: Apply it on undiscorted image
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    pts_left = np.array([np.transpose(np.vstack([left.x, left.y]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right.x, right.y])))])
    pts = np.hstack((pts_left, pts_right))
    cv2.fillPoly(color_warp, np.int_([pts]), (255,255, 0))
    # Inverse Transformation
    M_inv = transformer.perspective_transformer.M_inv
    new_warp = cv2.warpPerspective(color_warp, M_inv, (warped.shape[1], warped.shape[0]))
    # distort back
    new_img = cv2.addWeighted(img, 1, new_warp, 0.4, 0)
    # annotate curvature
    mean_curvature = np.mean([left.radius_of_curvature, right.radius_of_curvature])
    _str = "radius of curvature: {} km".format(round(mean_curvature / 1e3,2))
    cv2.putText(new_img, _str, (10,30), FONT, 1,(255,255,255), 2, cv2.LINE_AA)
    # vehicle distance from the center of the image
    _mid = abs((img.shape[1] / 2) - (left.base_position + right.base_position) / 2) * left.xm_per_pix
    _str = "distance from center: {} cm".format(round(_mid, 3))
    cv2.putText(new_img,_str, (10,60), FONT, 1,(255,255,255), 2, cv2.LINE_AA)
    #vehicle_offset = (left.xm_per_pix * (img.shape[1] / 2) - _mid
    return new_img

def process_image(img, transformer, left, right):
    warped_image = transformer.transform(img)
    left_points, right_points = detect_points(warped_image)
    # update
    left.process_image(left_points)
    right.process_image(left_points)
    # draw
    # Recast the x and y points into usable format for cv2.fillPoly()
    new_img = overlay_detected_lane(img, transformer, warped_image, left, right)
    return new_img

if __name__ == '__main__':

    in_file = arguments['<input_video>']
    out_file = arguments['<output_video>']


    print("Prepare the transformation pipeline for image preprocessing")
    transformer = setup_transformation_pipeline()
    warp = partial(process_image, transformer = transformer, left = Line(), right = Line())

    print('Processing video ...')
    clip2 = VideoFileClip(in_file)

    vid_clip = clip2.fl_image(warp)
    vid_clip.write_videofile(out_file, audio=False)
