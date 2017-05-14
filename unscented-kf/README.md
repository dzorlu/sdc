# Unscented Kalman Filter Project
C++ implementation of an unscented kalman filter. 

UKFs approximate process and measurement transformations through the use of sigma points.    

---

## Dependencies

* cmake >= v3.5
* make >= v4.1
* gcc/g++ >= v5.4

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./UnscentedKF path/to/input.txt path/to/output.txt`. You can find


## Reported RMSE 

    0.0641554
    0.0877843
    0.336986
    0.224262

## NIS

![Lidar](img/lidar_nis.png)

![Ridar](img/radar_nis.png)
 
 