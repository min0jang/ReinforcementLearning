
# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_images/center.png
[image2]: ./writeup_images/left.png
[image3]: ./writeup_images/right.png
[image4]: ./writeup_images/flipped.png
[image5]: ./writeup_images/model.png

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is written with keras, and I used the model designed by nvdia, that was specifically designed for the steering of a self driving car: https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/. 

#### 2. Attempts to reduce overfitting in the model
The 'Dropout' layer is in fact not part of the original design by nvdia, but I added it in an attempt to reduce the possibility of overfitting during the training process.

In the beginning of the 'model.py', I divided the sample data into two segments: training set(80%) and validation set(20%). This made the model to be trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.


#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.


#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. It was important to a set of meaningful data set that can actually train the model how to bounce back to the center line when it is too close to the boundaries. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to try pre-certified models such as LeNet and nvdia model and choose the best-functioning one.

After trying various models, I chose the nvdia model. I thought this model might be appropriate because it is designed by great engineers at nvdia with the specific goal to be the same as mine in this project.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that it has a dropout layer.

Then I saw that the model was no longer overfitting.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes.

![alt text][image5]



#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to come back to the center line. For the corresponding steer angle values, I added and subtracted certain correction factor, given that the photos were taken by the left and right cameras, not the center. These images show what a recovery looks like starting from left and right :

-LEFT-
![alt text][image2]

-RIGHT-
![alt text][image3]

To augment the data sat, I also flipped images and angles thinking that this would enrich the training set. The track I used for attaining sample data was mostly composed of left-turning corners, which would weaken the model's reaction to right-turning corners. For example, here is an image that has then been flipped:

![alt text][image4]


After the collection process, I had over 23000  data points. I then preprocessed this data by cropping the unnecessary parts and normalize the color values to have zero-mean and low standard deviation.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by the loss graph of the training process. The graph showed that the loss fluctuates up and down after the fifth epoch. I used an adam optimizer so that manually training the learning rate wasn't necessary.

