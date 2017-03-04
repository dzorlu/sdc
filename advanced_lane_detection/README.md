

In this project, I used Kalman filters to track the lane lines after successful detection of lane pixels. Binary masking used to detect lane pixels consist of a main masking filter that is a combination of laplacian, saturation, and gray image masking combined with a secondary fallback filter that utilizes sobel thresholding.

The polynomial curvature and position of the lane lines in turn inform us about the curvature of the road and the distance of the center of the vehicle to the middle of lane line. The project is accompanied with three videos annonated by lane markings, curvature of the road, and the distance of the vehicle to the center of the road.


The steps followed are as follows:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").

* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.


## Camera Calibration
Images captured by a camera are typically distorted by the lense.  Using a distorted image would cause issues if one attempts to calculate statistics based on it. The first step in the lane detection pipeline is to undistort the image by computing the transformation between 3D object points in the world and 2D image points. Samples of chessboard patterns recorded from different angles are used to [calibrate](https://github.com/dzorlu/sdc/blob/master/advanced_lane_detection/image_transformation.py#L13) the camera in order to [undistort](https://github.com/dzorlu/sdc/blob/master/advanced_lane_detection/image_transformation.py#L40) the incoming images.

![Image before and after undistortion](https://github.com/dzorlu/sdc/blob/master/advanced_lane_detection/writeup_images/undistort.png)

## Lane Masking
As the second step, I tried several binary masking methods to compare the performance. I came back and modified my approach iteratively to create fallback options if primary binary image didn't capture left or right side of the point of view. Overall, I attempted 9 binary masking  [techniques](https://github.com/dzorlu/sdc/blob/master/advanced_lane_detection/image_transformation.py) all of which are shown below. My experience led to the belief that laplacian, saturation, and gray channels work the best.

![Masking Techniques](https://github.com/dzorlu/sdc/blob/master/advanced_lane_detection/writeup_images/masking.png)

It follows the following steps for image masking:

 - primary filter: combination of laplacian, saturation, and gray image masking

  `combined_binary = cv2.bitwise_and(laplacian_binary, cv2.bitwise_or(s_binary, gray_binary))`

 - secondary filter: sobel thresholding

   `x_y_binary = cv2.bitwise_and(x_binary, y_binary)`

 - [region of interest](https://github.com/dzorlu/sdc/blob/master/advanced_lane_detection/image_transformation.py#L111) filters ignore regions outside our scope and focus on the lower triangle of the image.

 Lane masking process couple with filtering the region of interest produces the following result.

 ![After Lane Masking and Region of Interest Filtering](https://github.com/dzorlu/sdc/blob/master/advanced_lane_detection/writeup_images/masked_image.png)


## Perspective transform
Next, we transform the perspective from head-one camera view to "bird's eye". We do this in order to (i) identity the lane more accurately (ii) compute the curvature of the road. The technique requires two points - source and destination - to define the transformation mapping. I used an image where the lane marking was straight and clearly marked to calibrate the perspective matrix with [PerspectiveTransformer](https://github.com/dzorlu/sdc/blob/master/advanced_lane_detection/image_transformation.py#L244).


Given the masked image, the destination and source points can be seen below, where red and blue dots and source and destination points, respectively. Note that the points overlap on the horizontal axis.

![Source and Destination](https://github.com/dzorlu/sdc/blob/master/advanced_lane_detection/writeup_images/perspective_transform.png)

The transformation produces a warped image.

![transformation](https://github.com/dzorlu/sdc/blob/master/advanced_lane_detection/writeup_images/perspective_transform2.png)

Finally I applied some extra filtering like [histogram filters](https://github.com/dzorlu/sdc/blob/master/advanced_lane_detection/image_transformation.py#L168). Histogram filters are another layer to eliminate the outliers in the image by computing the pixel intensity along the horizontal axis and accepting pixels that are in the vicinity of the peak for right and left halves of the image.

`TransformationPipeline` succinctly implements the pipelining discussed so far. Input is the incoming video frame and the output is the warped image that is undistorted, masked, warped, and filtered.

```
class TransformationPipeline():
    def __init__(self, camera_calibration, src, dst ):
        self.camera_calibration = camera_calibration
        self.perspective_transformer = PerspectiveTransformer(src, dst)

    def transform(self, img):
        _img = self.undistort_image(img)
        # depending on the avail of filtered_warped_image apply another round of masking
        binary_img = self.lane_masking(_img)
        warped_image = self.perspective_transform(binary_img)
        filtered_warped_image = self.histogram_filter(warped_image)

        return filtered_warped_image

    def undistort_image(self, img):
        return undistort_image(img, self.camera_calibration)

    def lane_masking(self, img):
        return lane_masking(img)

    def post_lane_masking(self, img, warped):
        return post_lane_masking(img, warped)

    def region_of_interest(self, img):
        # Filters the image for the lower trapezoid
        return region_of_interest(img)

    def perspective_transform(self, img):
        return self.perspective_transformer.transform(img)

    def inverse_perspective_transform(self, img):
        return self.perspective_transformer.inverse_transform(img)

    def histogram_filter(self, img):
        return histogram_filter(img)
```

## Lane Tracking Pipeline

I keep track of the fitted polynomial line using Kalman filters. All properties of the line are tracked within an instance of the [`Line`](https://github.com/dzorlu/sdc/blob/master/advanced_lane_detection/lane_detection.py#L42) class. It should be noted that we keep track of left lane and right lane separately throughout.

The Kalman filters consist of prediction and updates states. In the update state, we take a measurement, which in this case is the lane pixel points detected in the image. Intuitively, the update state reduces uncertainty about whereabouts of the object. But the passage of time and the randomness of the motion of the the object we are tracking introduce uncertainty. I have found that this approach is applicable to the problem because of two reasons:

 - Pixels detected in the current image contain some measurement noise, i.e. we are not certain that the observation identifies the lane lines with full accuracy. We need to quantity how much belief we should have on evidence versus the priors.
 - There are cases where the `TransformationPipeline` fails to return any lane pixels. In such cases, we need to account for the fact that we are facing an uncertain world and the vehicle might not be where we detected the last time - several frames back. Hence we need to inject some uncertainty into the current state of lane object we are tracking.

Before I discuss the methodology in more detail, I want to talk about the lane tracking pipeline. Lane tracking update is performed with the [`process_image`](https://github.com/dzorlu/sdc/blob/master/advanced_lane_detection/lane_detection.py#L128) method shown below.

Processing step takes in pixel points detected in the transformation pipeline and fits a polynomial function with `self.fit_poly_lanes()` to identify the curved line. Because the lane lines in the warped image are near vertical and may have the same x value for more than one y value. This means that the `self.set_fitted_x()` method uses static y-axis to predict the x-value for a given lane pixel located in (x,y) coordinate.

```
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
```

Next, we update the Kalman filter state. The state

the weighted average of prediction and measurement is based on variances. The more confidence you have on your priors, it will be more difficult to move the mean. `kalman_gain` x `_residual` gives you the adjustment in pixels. assumes  It should also be noted that the Kalman filter is an one-dimensional filter because we assume a constant velocity for the vehicle.

[]()


  If the point is too far off, the `Line` rejects the proposed image and moves on to the next image. Update Kalman filter state even though there are no detected lines by injecting some noise into the state and incrementing the state noise. This approach enables a quicker adjustment to the next detected lane.

Kalman filter is initialized such that it takes 5 seconds for the filter completely adjust to the signal. The video streams at a 25 pixels per second. That means the state gets updated 25 times and kalman update


http://www.intmath.com/applications-differentiation/8-radius-curvature.php
