# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./readme_images/01_nvidia_arch.png "nVidia architecture"
[image2]: ./readme_images/02_LeNet.png "LeNet architecture"
[image3]: ./readme_images/03_ModelArch.png "Final architecture"
[image4]: ./readme_images/04_Centered_-0.01408.jpg "Centered"
[image5]: ./readme_images/05_SharpCornerTrack1_-0.21127.jpg "Sharp Corner 1 Track 1"
[image6]: ./readme_images/06_SharpCorner2Track1_0.211268.jpg "Sharp Corner 2 Track 1"
[image7]: ./readme_images/07_LeftRecoveryTrack1_0.183099.jpg "Left Recovery Track 1"
[image8]: ./readme_images/08_LeftRecoveryTrack1_0.15493.jpg "Left Recovery Track 1"
[image9]: ./readme_images/09_LeftRecoveryTrack1_-0.08451.jpg "Left Recovery Track 1"
[image10]: ./readme_images/10_BridgeTrack1_0.004695.jpg "Bridge Track 1"
[image11]: ./readme_images/11_StraightTrack2_-0.02817.jpg "Centered Track 2"
[image12]: ./readme_images/12_SharpCornerTrack2_1.jpg "Sharp Corner Track 2"
[image13]: ./readme_images/13_SharpCornerTrack2_0.9624.jpg "Sharp Corner Track 2"
[image14]: ./readme_images/14_SharpCornerTrack2_0.831.jpg "Sharp Corner Track 2"



## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* README.md (this document) summarizing the results
* video.mp4 video showing the car driving around track 1 in autonomous mode
* track2.mp4 video showing the car driving around track 2 in autonomous mode

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is based on the [nVidia architecture](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)

![alt text][image1]

One difference is that in this project, captured images are of the size 320x160x3, not the 200x66x3 that are used in the paper.

First, the images are normalized using Keras Lambda layer by dividing pixel data by 255 and subtracting 0.5. (model.py line 101)
Then, images are cropped using Cropping2D layer in Keras. (model.py line 102)

After that the model follows the same structure as the nVidia model:
* Three convolutional layers of depth 24, 36, and 48. These layers use a 5x5 kernel size and a 2x2 stride. (model.py lines 103-105)
* Two convolutional layers of depth 64. These use a 3x3 kernel size and a 1x1 stride. (model.py lines 106-107)
* Flattening layer (model.py line 108)
* Four fully connected layers with 100, 50, 10, and 1 neurons. (model.py lines 109-112)

The model includes RELU activation in the convolutional layers to introduce nonlinearity. (model.py lines 103-107)

Adam optimizer was chosen and mean squared error was used as a loss function. (model.py line 113)

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting (model.py line 28).
Dropout is not mentioned in the nVidia paper so it was not used in this model. A diverse set of data was used to make sure that model was flexible enough to perform well on both test tracks. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track on both tracks.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 113).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving well on both test tracks and being able to recover from difficult situations.

I've used my own data instead of relying on provided sample data from Udacity. I decided to do this because I would be able to control the distribution of data between various driving scenarios better this way.
I used a combination of center lane driving, sharp left and right turns, recovering from the left and right sides of the road, driving on both test tracks in forward and reverse direction, and recording of extra data for challenging corners. 

For details about how I created the training data, see the next section. 

### Architecture and Training Documentation

#### 1. Initial Model Architecture

The overall strategy for deriving a model architecture was to start of with something simple, make sure that my code and simulator worked as expected, and then move on to a more complicated model.

My first step was to use a convolution neural network model similar to the LeNet architecture. 
I thought this model might be an appropriate starting point because this model worked well enough as a starting point for traffic sign recognition project. The problem is different, but I wanted to just get something working first before moving on to a more complicated model. 
Below is an image of LeNet architecture as applied to traffic sign image input with dimensions of 32x32. 

![alt text][image2]

I kept the architecture the same, just the input dimensions were different at 320x160x3. Also, the normalization and cropping was applied to the input images. I've also only used the central camera images and the flipped images. Data was only collected on the initial test track and driving was done in the forward direction.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. 
I trained the model for 5 epochs as that seemed to be the point at which the model started to overfit as the validation set loss would increase after that point.

The final step was to run the simulator to see how well the car was driving around track one. To my surprise, it was actually able to drive around most of the track but it did so with a lot of wiggling and a few times it went off track, mostly around sharp corners. 

At this point, I was satisfied with the results as I knew that I needed to collect more data and create a more sophisticated model architecture to achieve better results.

#### 2. Final Model Architecture

The model that I chose to actually use for this project was based on the nVidia's paper mentioned above as it has been shown to work well for this project based on experiences of other students.

The final model architecture consisted of a convolutional neural network with the following layers:

* Normalization layer (model.py line 101)
* Cropping layer (model.py line 102)
* Three convolutional layers of depth 24, 36, and 48. These layers use a 5x5 kernel size and a 2x2 stride. (model.py lines 103-105)
* Two convolutional layers of depth 64. These use a 3x3 kernel size and a 1x1 stride. (model.py lines 106-107)
* Flattening layer (model.py line 108)
* Four fully connected layers with 100, 50, 10, and 1 neurons. (model.py lines 109-112)

Here is a visualization of the architecture:

![alt text][image3]


#### 3. Creation of the Training Set & Training Process

Selecting an appropriate neural network architecture for the task is important. Have a good data set to train the model on is equally as important as I've learned from the previous project. That's why I've decided to collect my own driving data instead of using pre-existing data set. This way I could control how much data of each driving behavior I would use to train and validate the model.

Majority of driving around the track focuses on keeping the car positioned around the middle of the lane, so I've recorded a lap of driving around the track forwards and backwards, while trying to keep the car as close to the middle of the road as possible.

Here is an example image of driving straight on the first track:

![alt text][image4]

The steering wheel angle here was just 0.35 degrees to the left, so almost straight.

At this point, from driving around track 1, I knew that it only has two sharper corners that require larger steering angles. So I've recorded taking them a few extra times, to make sure the model would have enough data to learn how to take those.

Here's an example of the first one:

![alt text][image5]

Steering wheel angle here was 5.28 degrees to the left.

And an example of the second one, which comes shortly after the previous corner.

![alt text][image6]

Steering wheel angle here was also 5.28 degrees but this time to the right.


I then recorded a number of recovery attempts, where the vehicle was about to head off track and I would pull it back to the center by applying a large streeing angle in the opposite direction. This would teach the model to be able to recover in case the vehicle ended up close to the edge of the track. These images show what a recovery looks like starting from the left edge of the track :

![alt text][image7]

![alt text][image8]

![alt text][image9]

I also recorded some more examples of driving on the bridge on track 1 as the road texture was different there from the rest of the road surface. This should allow the model not get confused by this. Here's an example of driving straight over the bridge:

![alt text][image10]

I've also recorded recovery attemps of when the vehicle was about to hit the side wall of the bridge, since previous recoveries were done on a different driving surface.

In an attempt to get a balanced data set of steering angles corresponding to left and right corners, I've recorded a lap around track 1 driving in the opposite direction. The idea is that if the original track has mostly left corners, driving in the opposite direction would result in mostly right corners, so the model would not be overly trained to turn left.


I then moved on to track 2 to allow the model to learn how to drive on a completely different track. This track is a lot more challenging and has corners that are much sharper than track 1, so it took a little bit of practice to get good data as I was struggling to keep the vehicle turns as smooth as possible. After I drove around the track once, I drove around it in reverse direction in order to get more diverse data set.

Here's an example of driving straight on track 2:

![alt text][image11]


The images shown above are recorded from the camera positioned in the center of the vehicle. However, at each time step two more images were being captured: one from the left mounted and one from the right mounted camera. I've used those images by associating an adjusted steering angle, which was the center camera's steering angle with a correction factor applied to it. For the left camera image a correction factor of 0.3 was added to the center camera's steering angle. For the right camera image a correction factor of 0.3 was subtracted from the center camera's steering angle. (model.py lines 78 and 79).
I've also flipped the center camera image and flipped the associated steering angle value to get even more data out of the recorded images. (model.py line 77).

Before training, I've split the data into training and validation data, allocating 20% of the data into validation set.
The validation set helped determine if the model was over or under fitting. 
I also randomly shuffle each data set as a part of the generator function (model.py line 40), which was used to reduce memory requirements (model.py lines 37-89).

The ideal number of epochs was 3 as evidenced by the loss function starting to increase after further epochs. 
I used an adam optimizer so that manually training the learning rate wasn't necessary.

After training the model using all of this data, the car was able to drive autonomously around track 1 without going out of bounds, but it had problems taking some of the sharper corners on track 2.

To combat that, I've gone back and recorded more data for specific corners that I saw were a struggle. Below is an example of one such corner:

![alt text][image12]

![alt text][image13]

![alt text][image14]

At this corner the steering wheel angle was at its maximum of 25 degrees to the right.

Collecting this additional data and training the model again resulted in the vehicle being able to successfully drive around track 2 in autonomous mode.

My final data set, including all of the scenarios and cases listed above, had a total of 50700 data points.

Overall, this was a very fun project, which highlighted how important it is to have good data for a given task.

