# Detecting Lane Lines

In this project, I used Kalman filters to track the lane lines after successful detection of lane pixels. Binary masking used to detect lane pixels consists of a main masking filter combined with a secondary fallback filter. Kalman filters keep track of fitted lane points on both sides of the car.

The output metadata - the polynomial curvature and position of the lane lines- in turn informs us about the curvature of the road and the distance of the center of the vehicle to the middle of lane line. The project is accompanied with three videos annonated by lane markings, curvature of the road, and the distance of the vehicle to the center of the road.


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
As the second step, I tried several binary masking methods. Lane masking is required to detect the location of the lane lines. I came back and modified my approach iteratively by creating a fallback option. If primary binary image failed to capture the left or right side of the point of view, secondary filter is applied to the image. Additionally, the lane masking method only accepts pixels that are within the region of interest.

Overall, I attempted 9 binary masking  [techniques](https://github.com/dzorlu/sdc/blob/master/advanced_lane_detection/image_transformation.py) all of which are shown below. After inspection on multiple images, I concluded that the laplacian, saturation, and gray channels work the best.

![Masking Techniques](https://github.com/dzorlu/sdc/blob/master/advanced_lane_detection/writeup_images/masking.png)

The module consists of following steps:

 - primary filter: combination of laplacian, saturation, and gray image masking

  `combined_binary = cv2.bitwise_and(laplacian_binary, cv2.bitwise_or(s_binary, gray_binary))`

 - secondary filter: sobel thresholding

   `x_y_binary = cv2.bitwise_and(x_binary, y_binary)`

 - [region of interest](https://github.com/dzorlu/sdc/blob/master/advanced_lane_detection/image_transformation.py#L111) filters ignore regions outside our scope and focus on the lower triangle of the image.


Lane masking process couple with filtering the region of interest produces the following result.

 ![After Lane Masking and Region of Interest Filtering](https://github.com/dzorlu/sdc/blob/master/advanced_lane_detection/writeup_images/masked_image.png)


## Perspective transform
Next, we transform the perspective from head-one camera view to "bird's eye". We do this in order to (i) identity the lane more accurately (ii) compute the curvature of the road. The technique requires two points - source and destination - to define the transformation mapping. In order to calibrate the perspective matrix, I used an image where the lane marking was straight and clearly marked.  [PerspectiveTransformer](https://github.com/dzorlu/sdc/blob/master/advanced_lane_detection/image_transformation.py#L244) is the class that implements the logic.


Given the masked image, the destination and source points can be seen below, where red and blue dots are source abd destination points, respectively. Note that the points overlap on the horizontal axis.

![Source and Destination](https://github.com/dzorlu/sdc/blob/master/advanced_lane_detection/writeup_images/perspective_transform.png)

The transformation produces a warped image.

![transformation](https://github.com/dzorlu/sdc/blob/master/advanced_lane_detection/writeup_images/perspective_transform2.png)

Finally I applied some extra [filtering](https://github.com/dzorlu/sdc/blob/master/advanced_lane_detection/image_transformation.py#L168). Histogram filter computes the pixel intensity along the horizontal axis and accepts pixels that are in the vicinity of the peak for right and left halves of the image.

`TransformationPipeline` succinctly implements the pipeline discussed so far with its method `transform`. Input is the incoming video frame and the output is the warped image that is undistorted, masked, warped, and filtered.

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

I keep track of the points on the fitted polynomial lines using Kalman filters. All properties of the lines are tracked within an instance of the [`Line`](https://github.com/dzorlu/sdc/blob/master/advanced_lane_detection/lane_detection.py#L42) class, which is a child a [Kalman filter class](https://github.com/dzorlu/sdc/blob/master/advanced_lane_detection/kalman_filter.py). It should be noted that we keep track of left lane and right lane separately throughout.

The Kalman filters consist of prediction and updates states. In the update state, we take a measurement, which in this case is the new lane pixel points detected in the image. Intuitively, the update state reduces uncertainty about whereabouts of the object. But the passage of time and the randomness of the motion of the the tracked objects introduce uncertainty. I have found that this approach is applicable to the problem because of two reasons:

 - Pixels detected in the current image contain some measurement noise, i.e. we are not certain that the observation identifies the lane lines with full accuracy. We need to quantity how much belief we should have on evidence versus the priors.
 - There are cases where the `TransformationPipeline` fails to return any lane pixels. In such cases, we need to account for the fact that we are facing an uncertain world and the vehicle might not be where we detected the last time - several frames back. Hence we need to inject some uncertainty into the current state of lane object we are tracking.

If the image did not produce any pixels to work on, the `process_image` method proceeds with the predict step.

Before I discuss the methodology in more detail, I want to talk about the lane tracking pipeline. Lane tracking update is performed with the [`process_image`](https://github.com/dzorlu/sdc/blob/master/advanced_lane_detection/lane_detection.py#L128) method shown below.

Processing step takes in pixel points detected in the transformation pipeline and fits a polynomial function with `self.fit_poly_lanes()` to identify the curved line. Because the lane lines in the warped image are near vertical and may have the same x value for more than one y value, the `self.set_fitted_x()` method uses static y-axis to predict the x-values for a given lane pixel located in (x,y) coordinate. `self.evaluate()` is another filter that rejects polynomial fits that are too far away from the state of the Kalman filter.

```
def process_image(self, pts):
    self.pts = pts
    self.increment_detection_count()
    if len(self.pts) > 0:
        self.fit_poly_lanes()
        self.set_fitted_x()
        # Reject if incoming polynomial fit is too far away
        if self.evaluate():
            # kalman update baseline next step
            self.update(self.x)
            # Using updated step, calculate the following:
            # curvature update
            self.set_curve_radius()
            # base position update
            self.set_base_position()
    else:
        # If no points found, predict the next step
        self.predict()
```

Next, we update the filter state with `self.update()`. We could be tracking several things including the polynomial coefficients or the points in the image. For this project, I decided to track the points instead of the polynomial fit because I find it more intuitive. It should also be noted that the Kalman filter is an one-dimensional filter because we assume a constant velocity for the vehicle. Secondly, I assume independence between the points I track. This is a gross simplification because the neighboring points in a lane are dependent on each other.  Third, because the points in the horizon should move faster, the measurement updates in the filter carry more weight. In Kalman filter world, faster adjustment translates into a larger Kalman gain, which is defined as the ratio of state noise to statement noise and measurement noise combined.

`_kalman_gain = self.state_noise / (self.state_noise + self.measurement_noise)`

A lower measurement noise returns a higher kalman gain, which pulls the current state more towards its direction. The procedure is as follows:

```
_residual = update - self.s
self.s = _kalman_gain * _residual + self.s
```

We have to decide on the parameters for the filter. The video streams at a 25 pixels per second. Kalman filter is initialized such that it takes 1 second for the filter completely adjust to the measurement difference of 25 pixels. For example, given the previous state of the pixel at x = 200 and the constant measurement at x = 225, the lane tracking object will take 1 second to shift the state of the pixel point completely. Pixels in the horizon adjust twice faster and the adjustment speed increases monotonically from the front of the car out to the horizon. This means that we have lower measurement noise as we go out to horizon. The implementation details of the filter can be found in [kalman_filter.py](https://github.com/dzorlu/sdc/blob/master/advanced_lane_detection/kalman_filter.py)

In the update step, we inject some uncertainty into the state information and increment the noise of the process. In all, after many prediction and update steps, the state variance should converge to a number that is satisfactory for our purposes.

Kalman filters ensure smooth averaging over many pixel instances through a Bayesian update mechanism. In addition, the filters allow us to factor in more uncertainty over our beliefs if we fail to detect the lane lines over multiple instances.


## Determining curvature and vehicle position with respect to center

The radius and curvature is computed in [`set_curve_radius`](https://github.com/dzorlu/sdc/blob/master/advanced_lane_detection/lane_detection.py#L77).

```
def set_curve_radius(self, order=2):
    """Curve Radius Based on Smoothed Lane Lines"""
    fit_cr = np.polyfit(self.y * self.ym_per_pix, self.state * self.xm_per_pix, order)
    y_max = np.max(self.y)
    curverad = ((1 + (2*fit_cr[0]*y_max + fit_cr[1])**2)**1.5) / abs(2*fit_cr[0])
    self.radius_of_curvature = curverad
```

The [vehicle position](https://github.com/dzorlu/sdc/blob/master/advanced_lane_detection/lane_detection.py#L145) with respect to center is computed as the distance of mean of both left and right base points to the center of the image, which in our case is assumed to be the center of the vehicle.

In both cases, we make sure that distances in terms of pixels are translated into real world distances.
```
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension
```
## Warping the detected lanes back to original image
Finally, we warp back the image back to the its original viewpoint. [`overlay_detected_lane`](https://github.com/dzorlu/sdc/blob/master/advanced_lane_detection/lane_detection.py#L125) method does the transformation, finalizes the radius and distance calculations and annotates the output image along with the lane boundaries.

The steps described above produces the final output for a given image.
![Image](https://github.com/dzorlu/sdc/blob/master/advanced_lane_detection/writeup_images/output.png)

# Summary and Results
The complete transformation pipeline is run via [`process`](https://github.com/dzorlu/sdc/blob/master/advanced_lane_detection/lane_detection.py#L153) helper method.

```
warped_image = transformer.transform(img)
# separete points into left and right
left_points, right_points = identify_points(warped_image)
# update step. process each half separately.
left.process_image(left_points)
right.process_image(right_points)
# draw - recast the x and y points into usable format for cv2.fillPoly()
# annotate the image with metadata
new_img = overlay_detected_lane(img, transformer, warped_image, left, right)
```

[Project Video](https://github.com/dzorlu/sdc/blob/master/advanced_lane_detection/annotated_project_video.mp4)

[Challenge Video](https://github.com/dzorlu/sdc/blob/master/advanced_lane_detection/annotated_challange_video.mp4)

[Harder Challenge Video](https://github.com/dzorlu/sdc/blob/master/advanced_lane_detection/annotated_harder_challange_video.mp4)

This is an initial version of advanced computer-vision-based lane finding. The lane finder still has hard time with the challenge videos, where lighting angles or the presence of the cars can easily trick the system. In situations where the road curves aggressively also turns out to be problematic for the lane finder. More work needs to be done to both improve the lane detection and calibration of the filter to obtain more robust results.
