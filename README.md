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

[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

[image1]: ./readme_images/01_nvidia_arch.png "nVidia architecture"
[image2]: ./readme_images/02_LeNet.png "LeNet architecture"
[image3]: ./readme_images/03_ModelArch.png "Final architecture"

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
* Flattening layer
* Four fully connected layers with 100, 50, 10, and 1 neurons.

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

### Model Architecture and Training Strategy

#### 1. Initial Model Architecture

The overall strategy for deriving a model architecture was to start of with something simple, make sure that my code and simulator worked as expected, and then move on to a more complicated model.

My first step was to use a convolution neural network model similar to the LeNet architecture. 
I thought this model might be an appropriate starting point because this model worked well enough as a starting point for traffic sign recognition project. The problem is of course different, but I wanted to just get something working first. 
Below is an image of LeNet architecture as applied to traffic sign image input with dimensions of 32x32. 

![alt text][image2]

I kept the architecture the same, just the input dimensions were different at 320x160x3. Also, image inputs were normalized using the Lambda layer, but no cropping has been done yet. I've also only used the central camera images and the flipped images. Data was only collected on the initial test track and driving was done in the forward direction.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. 
I trained the model for 5 epochs as that seemed to be the point at which the model started to overfit as the validation set loss would increase after that point.

The final step was to run the simulator to see how well the car was driving around track one. To my surprise, it was actually able to drive around most of the track but it did so with a lot of wiggling and a few times it went off track, mostly at sharp corners. 

At this point, I was satisfied with the results as I knew that I needed to collect more data and create a more sophisticated model architecture to achieve better results.

#### 2. Final Model Architecture


The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture:

![alt text][image3]


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.

### Conclusion and Discussion