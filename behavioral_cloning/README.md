#Behavioral Cloning

The goal of the exercise is to get a car to drive autonomously in a simulated setting. Udacity provides the simulator, which includes two tracks that the model can be trained on. First track is easier. Second track includes hills, tight turns, shadows on the road, bridges etc, all of which makes it harder to navigate.

Although Udacity provides the [dataset](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip) to get something working off the bat, I quickly realized that I not only need to collect more data through the simulator, but also augment and preprocess the data in smart ways to keep the car on the track and out of harms way.

In supervised learning, it is assumed that test data comes from the same distribution of training data. The assumption is violated in this exercise, however, because the errors will be propagated onto the next sequences as the autonomous car drives on. This might result in errors compounding and expose the car to instances that was not present in the training set. This problem is simply circumvented by recording instances from a left and right camera to simulate recovery instances. The same approach is adopted in the exercise by default.


##The Model

There is nothing novel about the model I have used. The model normalizes the input layer akin to the normalization layer in the [NVIDIA](https://arxiv.org/abs/1604.07316) model. It is a standard CNN that has three convolutional layers, each of which followed by 2x2 max pooling. Convolutional layers are followed by three fully connected layers. Because the fully connected layers have a large capacity, L2 norm is used to battle overfitting. Activation layers are all RELU layers. I used Adam optimizer, which attempts to apply best of both worlds - momentum and varying learning rates for each parameter. Because the prediction is the steering angle of the car, the error rate is defined as mean squared error.

```
model = Sequential()
model.add(Lambda(lambda x: x/255.-0.5,input_shape=self.input_shape))
model.add(Convolution2D(16, f1, f1, input_shape=(32, 128, 3), activation='relu', border_mode='same'))
model.add(MaxPooling2D(pool_size=self.pool_size))
model.add(Convolution2D(32, f2, f2, activation='relu', border_mode='same'))
model.add(MaxPooling2D(pool_size=self.pool_size))
model.add(Convolution2D(64, f3, f3, activation='relu', border_mode='same'))
model.add(MaxPooling2D(pool_size=self.pool_size))
model.add(Convolution2D(64, f3, f3, activation='relu', border_mode='same'))
model.add(MaxPooling2D(pool_size=self.pool_size))
model.add(Flatten())
model.add(Dropout(dropout_p))
model.add(Dense(self.n_fc, W_regularizer=l2(self.l2_reg), activation='relu'))
model.add(Dense(self.n_fc//2, W_regularizer=l2(self.l2_reg), activation='relu'))
model.add(Dense(1))
```

The model is trained over 75 epochs. In each epoch, I generated 10000 images. The batch size is 256.

The code details can be found under the `model` folder.

##Data Generation

The real meat of the project lies in data generation. In training mode, the simulator produces three images per each frame corresponding to right-, left-, and center-mounted cameras. Inclusion of right and left cameras is a common trick used to teach the car to swirl back to center the road as described above. In order to teach the model to swirl back to the center of the road, the recorded steering angle is adjusted appropriately. Additionally, I recorded extra data to provide data instances when the car is close to going off the road. By providing 'recovery' instances, we try to capture data on the tails of the steering angle distribution.

The challange is that the errors are propagated.

Images with ground truth values where steering angle is zero is overpresented in the dataset. This might be a problem in training such that the car might be unresponsive to turns. I chose to tackle this issue by smoothing the steering angle, mostly because my input into the simulator is not perfect. For example, I often have found myself letting go the wheel when the car is to much on one side. With smoothing, I attempt to make the ground truth simply more accurate to perfect driving.

I cropped off parts of the image that is not needed - roughly the bottom quartile and the top quartile. The brightness of the image is randomly distorted to provide a more diverse set of images. The images are symmetrical on the horizontal axis. That means we can double the size of the dataset simply by flipping the images left to right and vice versa. The adjustment happens to alleviate the bias in the training data somewhat. Last but not least, I applied some Gaussian smoothing. Last but not least, the second track is particularly hilly. The hilliness is manifested in the image data akin to some perspective transformation. I introduced perspective transformation as part of the transformation pipeline to be able to account for this phenomenon. The data generator closely resembles Keras. The reason why I built a new one instead of expanding the class is because Keras generator does not seem to augment the target variables along with feature variables.

In all, the training data has close to 50,000 images including left and right camera angles.

The code details can be found under the `data_generator` folder.

For more details please see the [iPython notebook](https://github.com/dzorlu/sdc/blob/master/behavioral_cloning/Behavioral%20Cloning.ipynb)

# What didn't work
I tried to detect the lanes using canny edge detection which didn't yield any improvements. In general, hyperparameter tuning to improve the model did not matter as much as gathering more data.


#Test Results
[![Easy Road](https://youtu.be/cegzGI5gkuw)
