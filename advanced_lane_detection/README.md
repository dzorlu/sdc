

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

## Lane Masking
As the second step, I tried several binary masking methods to compare the performance. I came back and modified my approach iteratively to create fallback options if primary binary image didn't capture left or right side of the point of view.



[]()
The steps are are as follows:
 - primary filter: combination of laplacian, saturation, and gray image masking

  `combined_binary = cv2.bitwise_and(laplacian_binary, cv2.bitwise_or(s_binary, gray_binary))`

 - secondary filter: sobel thresholding

   `x_y_binary = cv2.bitwise_and(x_binary, y_binary)`


 - region of interest filters to focus on the lower triangle of the image.


## Perspective transform
Perspective transform requires two points - source and destination - to define the transformation mapping. I used a straight pattern lane to calibrate the perspective matrix. `PerspectiveTransformer`

Finally I applied some extra filtering like histogram filters. I sample from the peak of the horizontal pixel intensity to filter out outliers.

## Finding the curvature

I keep track of the line using Kalman filters and all properties of the line are tracked within an instance of a `Line` class.

Kalman filters. assumes  a constant velocity for the car

The Kalman filter briefly consists of prediction and updates states. in the update state, the weighted average of prediction and measurement is based on variances. The more confidence you have on your priors, it will be more difficult to move the mean. `kalman_gain` x `_residual` gives you the adjustment in pixels.

[]()

`Line` is defined as a class that keeps track of lane elements using 1D Kalman Filters. It computes the radius of the curvature after fitting polynomial lanes and rejects outliers that are way too off from previous baseline pixel.

I first compute the polynomial of points detected in the current image. Because the lane lines in the warped image are near vertical and may have the same x value for more than one y value. The prediction for the baseline pixel (the pixel that sits on the horizontal axis) is used to update the state of the Kalman filter. If the point is too far off, the `Line` rejects the proposed image and moves on to the next image. Update Kalman filter state even though there are no detected lines by injecting some noise into the state and incrementing the state noise. This approach enables a quicker adjustment to the next detected lane.

Kalman filter is initialized such that it takes 5 seconds for the filter completely adjust to the signal. The video streams at a 25 pixels per second. That means the state gets updated 25 times and kalman update


http://www.intmath.com/applications-differentiation/8-radius-curvature.php
