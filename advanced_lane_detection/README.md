

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

I follows the following steps for image masking:

 - primary filter: combination of laplacian, saturation, and gray image masking

  `combined_binary = cv2.bitwise_and(laplacian_binary, cv2.bitwise_or(s_binary, gray_binary))`

 - secondary filter: sobel thresholding

   `x_y_binary = cv2.bitwise_and(x_binary, y_binary)`

 - [region of interest](https://github.com/dzorlu/sdc/blob/master/advanced_lane_detection/image_transformation.py#L111) filters ignores regions outside our scope and focus on the lower triangle of the image.

 Lane masking process in the pipeline gives me the following result.
 [After Lane Masking and Region of Interest Filtering](https://github.com/dzorlu/sdc/blob/master/advanced_lane_detection/writeup_images/masked_image.png)


## Perspective transform
Next, we transform the perspective from head-one camera view to "bird's eye". We do this in order to (i) identity the lane more accurately (ii) compute the curvature of the road. The technique requires two points - source and destination - to define the transformation mapping. I used an image where the lane marking was straight and clearly marked to calibrate the perspective matrix with [PerspectiveTransformer](https://github.com/dzorlu/sdc/blob/master/advanced_lane_detection/image_transformation.py#L244).

```
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
```

Given the masked image, the destination and source points can be seen below, where red and blue dots and source and destination points, respectively. Note that the points overlap on the horizontal axis.
[Source and Destination](https://github.com/dzorlu/sdc/blob/master/advanced_lane_detection/writeup_images/perspective_transform.png)

The result of the transformation for the same image
[transformation](https://github.com/dzorlu/sdc/blob/master/advanced_lane_detection/writeup_images/perspective_transform2.png)

Finally I applied some extra filtering like [histogram filters](https://github.com/dzorlu/sdc/blob/master/advanced_lane_detection/image_transformation.py#L168). Histogram filters are another layer to eliminate the outliers in the image by computing the pixel intensity along the horizontal axis and accepting filters that are closer to the peak for right and left lanes.

`TransformationPipeline`["https://github.com/dzorlu/sdc/blob/master/advanced_lane_detection/image_transformation.py#L258"] succinctly implements the pipelining discussed so far. 

## Finding the curvature

I keep track of the line using Kalman filters and all properties of the line are tracked within an instance of a `Line` class.

Kalman filters. assumes  a constant velocity for the car

The Kalman filter briefly consists of prediction and updates states. in the update state, the weighted average of prediction and measurement is based on variances. The more confidence you have on your priors, it will be more difficult to move the mean. `kalman_gain` x `_residual` gives you the adjustment in pixels.

[]()

`Line` is defined as a class that keeps track of lane elements using 1D Kalman Filters. It computes the radius of the curvature after fitting polynomial lanes and rejects outliers that are way too off from previous baseline pixel.

I first compute the polynomial of points detected in the current image. Because the lane lines in the warped image are near vertical and may have the same x value for more than one y value. The prediction for the baseline pixel (the pixel that sits on the horizontal axis) is used to update the state of the Kalman filter. If the point is too far off, the `Line` rejects the proposed image and moves on to the next image. Update Kalman filter state even though there are no detected lines by injecting some noise into the state and incrementing the state noise. This approach enables a quicker adjustment to the next detected lane.

Kalman filter is initialized such that it takes 5 seconds for the filter completely adjust to the signal. The video streams at a 25 pixels per second. That means the state gets updated 25 times and kalman update


http://www.intmath.com/applications-differentiation/8-radius-curvature.php
