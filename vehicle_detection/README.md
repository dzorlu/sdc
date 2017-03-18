# Vehicle Detection

In this project, I wrote a pipeline to detect vehicles in a video.

## Create a Feature Space and Train the Classifier
Training pipeline consists of generating HOG features and color histogram features followed by fitting a linear SVM model. I shuffle images to combat overfitting, which seems to be a problem because the dataset consists of identical / similar images. Last, the features are normalized both at training and test time.

Because the search pipeline uses an exhaustive search windowing each image, HOG features in combination with a linear classifier is the only feasible choice from a computational perspective.

```
vehicle_img_path, non_vehicle_img_path = read_images(12000)
print("Retrieved the image paths...")
print("{} images found".format(len(vehicle_img_path) + len(non_vehicle_img_path)))

## HOG Features
hog_features_vehicles = np.array([create_hog_features(_img) for _img in vehicle_img_path])
hog_features_non_vehicles = np.array([create_hog_features(_img) for _img in non_vehicle_img_path])
print("Created HOG Features...")

## Created Color Histograms
color_vehicles = np.array([create_color_hist(_img) for _img in vehicle_img_path])
color_non_vehicles = np.array([create_color_hist(_img) for _img in non_vehicle_img_path])
print("Created Color Histogram Features...")

vehicles = np.concatenate((hog_features_vehicles, color_vehicles),axis=1)
del hog_features_vehicles, color_vehicles
non_vehicles = np.concatenate((hog_features_non_vehicles, color_non_vehicles),axis=1)
del hog_features_non_vehicles, color_non_vehicles

X = np.vstack((vehicles, non_vehicles)).astype(np.float64)
y = np.hstack((np.ones(vehicles.shape[0]), np.zeros(non_vehicles.shape[0])))

print("Generated Feature Space {}".format(X.shape))
np.save("saved_models/X.npy", X)
np.save("saved_models/y.npy", y)
print("Saves Feature Space")
```

## Object Recognition

Similar to the last project, `Detector` class processes each image in following steps:

```
self.find_proposed_regions()
self.update_heatmap()
self.detect()
```

The search method first unleashes the trained linear SVM on the image using a sliding window at different shapes. In particular, I used four sliding windows with the following parameters. The larger filter with the largest scale scans the largest area. Each consecutive window is smaller and thereby scans a smaller region. `Y_CROP_TOP` parameter corresponds to the cutoff point for the Y-axis to the top.

```
self.scales = [3, 2, 2, 1]
self.cells_per_step = [1, 1, 1, 2]
self.y_crops = [(Y_CROP_TOP,650), (Y_CROP_TOP,600), (Y_CROP_TOP,550), (Y_CROP_TOP,500)]
```

I hand-tunes on the sliding window structure to accommodate enough number of sliding windows in each dimension. The parameters and the sub-image each window is processing can be seen below:

![Windows](https://github.com/dzorlu/sdc/blob/master/vehicle_detection/images/Screen%20Shot%202017-03-17%20at%202.30.53%20PM.png)



`Detector` is stateful and serves as a memory cell to track cumulative heatmap over consecutive frames. Here, I apply a decay function to update the heatmap with new information. In the second step, I apply a threshold to eliminate the false positives, which in turn is fed into [`label`](https://docs.scipy.org/doc/scipy-0.16.0/reference/generated/scipy.ndimage.measurements.label.html) method in order to identify independent object regions.

Because applying multiple sliding windows are expensive, I do this in a multi-threaded way. Further, `_find_proposed_regions` method computes the HOG features only once per image.
