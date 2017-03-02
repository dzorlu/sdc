


## Camera Calibration
Images captured by a camera are typically distorted by the lense. Using a distorted image would cause issues if one attempts to calculate statistics based on it. The first step in the lane detection pipeline is to undistort the input image where we compute the transformation between 3D object points in the world and 2D image points.

## Lane Masking
As the second step, I tried several methods that I outlined in the iPython notebook in order to detect the lane lines.


## Perspective transform
Perspective transform requires two points - source and destination - to define the transformation mapping.

## Finding the curvature


because the lane lines in the warped image are near vertical and may have the same x value for more than one y value.

all properties of the line are determined within an instance of a Line class

http://www.intmath.com/applications-differentiation/8-radius-curvature.php
