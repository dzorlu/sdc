# Vehicle Detection

In this project, I wrote a pipeline to detect and track vehicles in a video. Detection uses an ensemble classifier. Detector first produces proposals using a fast classifier - [Linear SVM](https://github.com/dzorlu/sdc/blob/master/vehicle_detection/src/train.py#L45)-. Then the proposals are evaluated by [a convolutional neural network](https://github.com/dzorlu/sdc/blob/master/vehicle_detection/src/train.py#L190) for validation. An ensemble method greatly reduces the frequency of false positives. The ensemble method passes the accepted proposals onto the heatmap, which masks accepted proposals by the intensity. The heatmap acts as a memory cell - keeping the detection in memory through weighted averages and only exposing areas that are _bright_ enough defined by some threshold.

## Create a Feature Space and Train the Classifier

Training the pipeline starts generating [HOG features](https://github.com/dzorlu/sdc/blob/master/vehicle_detection/src/preprocessing.py#L24) and spatial features followed by fitting a binary linear SVM model. I used all YUV channels to create the HOG features and hand-tuned the HOG parameters to get the best results. I used 4x4 spatial features. Other parameter values are given below.

```
ORIENT = 9 # number of bins
PIX_PER_CELL = 8
CELL_PER_BLOCK = 2
HOG_CHANNEL = [0,1,2] #use all channels in YUV spectrum.
```

I shuffle images to combat overfitting, which seems to be a problem because the dataset consists of identical / similar images. The data contains cars vs non-car images. I also included false-positives and false-positives that I identified at the test time, which helped a great deal. Last, the features are normalized both at training and test time.

Because the search pipeline uses an exhaustive search windowing each image, HOG features in combination with a linear classifier is the feasible choice from a computational perspective. The proposed regions by SVM is subsequently evaluated by a CNN. Ensemble methods generally work well because the uncorrelated errors between models cancel each other out, which leads to more robust predictions. In this particular case, the models evaluate different color spaces (YUV and BGR), which also helps with uncorrelated errors.

## Object Recognition

Similar to the last project, the [`Detector`](https://github.com/dzorlu/sdc/blob/master/vehicle_detection/src/search.py#L131) class processes each image and identifies regions that are likely to contain a car.

```
def process(self, img):
    self.nb_frames_processed += 1
    print(self.nb_frames_processed)
    self.image = img
    self.find_proposed_regions()
    self.update_heatmap()
    self.detect()
    if DEBUG:
        self.show_frame()
    return self.get_labels()
```

The search process applies the trained classifier on sub-regions of the image obtained through a sliding window technique. In particular, I used four sliding windows with the following parameters. Below, the filter with the largest scale scans the largest area. Each consecutive window is smaller and thereby scans a smaller region. `Y_CROP_TOP` parameter corresponds to the cutoff point for the Y-axis to the top.

```
self.scales = [3, 2, 2, 1]
self.cells_per_step = [1, 1, 1, 2]
self.y_crops = [(Y_CROP_TOP,650), (Y_CROP_TOP,600), (Y_CROP_TOP,550), (Y_CROP_TOP,500)]
```

I hand-tuned the structure to accommodate enough number of sliding windows across each dimension. You can see that smaller windows scan an area that is further out and smaller, because it is meant to detect the objects that are further out in the horizon and thus appear smaller.

![Windows](https://github.com/dzorlu/sdc/blob/master/vehicle_detection/images/Screen%20Shot%202017-03-17%20at%202.30.53%20PM.png)

In addition, the detector evaluates the SVM predictions in terms [confidence](http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC.decision_function) and accepts proposals that have high confidence scores in terms of distance to the hyperplane.

`Detector` is stateful and serves as a memory cell to track cumulative heatmap over consecutive frames. Here, I apply a decay function to update the heatmap with new information. Pixels that are proposed by multiple windows get assigned a higher weight through `power` parameter - I apply an exponential boosting factor to the number of detections for a given pixel.

Subsequently, I applied a threshold to eliminate the areas that are possibly false positives. Threshold evaluates the heatmap values and exposes regions that are _hot_ enough.  In turn, thresholded heatmap is fed into [`label`](https://docs.scipy.org/doc/scipy-0.16.0/reference/generated/scipy.ndimage.measurements.label.html) method in order to identify independent object regions.

Because applying multiple sliding windows are expensive, I do this in a multi-threaded way. To speed things further, `_find_proposed_regions` method computes the HOG features only once per image.

The pipeline produces the following - from left to right and top to bottom.

 i) region proposals
 ii) accepted regions
 iii) heatmap / memory
 iv) thresholded heatmap
 v) labels

![Pipeline](https://github.com/dzorlu/sdc/blob/master/vehicle_detection/writeup_images/pipe.png)

For more details please see the [Jupyter notebook](https://github.com/dzorlu/sdc/blob/master/vehicle_detection/vehicle_detection.ipynb).

## Results

[Vehicle Tracking](https://youtu.be/9cYdwWCOj0w)

## Next Steps
There is some additional work to be done in order to make the exposed regions more stable. Because parameters are hand-tuned, I suspect that too much of it would not generalize well to other videos. Secondly, I would like to try out state-of-art localization / object detection methods such as [R-CNN](https://arxiv.org/abs/1504.08083) or apply more classical object-tracking methods Kalman Filters etc.
