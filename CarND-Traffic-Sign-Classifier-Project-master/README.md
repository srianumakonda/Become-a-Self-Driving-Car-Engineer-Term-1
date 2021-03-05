# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

Number of training examples = 34799
Number of validation examples = 4410
Number of testing examples = 12630
Image data shape = (32, 32, 1)
Number of classes = 43

#### 2. Include an exploratory visualization of the dataset.

The dataset consists of 43 classes of road signs taken from the dataset. Here are some of the images. The code for displaying the images can be found in the 5th cell of the ipynb file.

![alt text](display.jpg)

```python
plt.tight_layout()
from random import randrange
fig,a = plt.subplots(4,10,figsize=(25,30))

for i in range(4):
    a[i][0].imshow(X_train[randrange(n_train)])
    a[i][1].imshow(X_train[randrange(n_train)])
    a[i][2].imshow(X_train[randrange(n_train)])
    a[i][3].imshow(X_train[randrange(n_train)])
    a[i][4].imshow(X_train[randrange(n_train)])
    a[i][5].imshow(X_train[randrange(n_train)])
    a[i][6].imshow(X_train[randrange(n_train)])
    a[i][7].imshow(X_train[randrange(n_train)])
    a[i][8].imshow(X_train[randrange(n_train)])
    a[i][9].imshow(X_train[randrange(n_train)])
```

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because I leanred that it would work really well with the LeNet architecture. I did actually play around with RGB images but they did not come out well mainly because the RGB channels consume a lot of memory, are computationally expensive, and it can result in higher variance. All of this resulted in lower accuracies.

I normalized the images by chosing to divide them by 255.0 instead of doing minmax and subracting 128 and then dividing. This is mainly because minmax values fall between 1 and -1 vs directly dividing them by 255 resulting in the pixel values to be between 0 and 1. This helps with having less variance and higher chances of being more accurate.

Here is an example of a traffic sign image before and after grayscaling:

![alt text](after.jpg)
![alt text](before.jpg)

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I used the LeNet model architecture provided but made some small modifications to it.

Here's a model summary:

Input size: (None,32,32,1)

Convolution - 5x5 kernel size, in_channel = 1, out_channel = 6
RelU
Maxpooling - 2x2 kernal size, stride = 2

Convolution - 5x5 kernel size, in_channel = 6, out_channel = 16
RelU
Maxpooling - 2x2 kernal size, stride = 2

Flatten layers into 1-D Vector
Feedforward Neural Network Layer - Input = 400, Ouptut = 1024
Feedforward Neural Network Layer - Input = 1024, Ouptut = 512

Output - Input = 512, Ouptut = 43


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

- Loss function: softmax_cross_entropy_with_logits
- Optimizer: Adam with a learning rate of 0.001 (1e-3)
- Batch size = 128
- Epochs = 20

I let this run for 20 epochs. Take a look below to see how well the model did!

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.9%
* validation set accuracy of 93.8% 
* test set accuracy of 92.5%

I increased the batch_size to 128 and increased epochs to 20 so that the model can train for longer. 

What I noticed was that after research, the drop from 84 nodes to 43 was too small of a delta. So what I did instead was that I actually increased the nodes from 400 to 1024. From there, I dropped the nodes to 512, and then straight to 43. That's how I got 99.9% accuracy.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text](1.jpg) ![alt text](2.jpg) ![alt text](3.jpg) 
![alt text](4.jpg) ![alt text](5.jpg)

Some of the images have terrible lighting (the 4th one) which can make it really difficult for the model to figure it out. 

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (30km/h)      		| Speed limit (30km/h)   									| 
| Slippery road     			| Slippery road 										|
| Turn left ahead					| Turn left ahead											|
| No passing for vehicles over 3.5 metric tons	      		| No passing for vehicles over 3.5 metric tons					 				|
| Speed limit (120km/h)			| Speed limit (120km/h)      							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%! I'm actually very surprised and proud of it, I thought the model would guess the 4th image wrong but it got it right!
#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 14th cell of the Ipython notebook.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Speed limit (30km/h)   									| 
| 1.0    				| Slippery road 										|
| 1.0 					| Turn left ahead											|
| 0.99	      			| No passing for vehicles over 3.5 metric tons					 				|
| 0.99				    | Speed limit (120km/h)      							|


I can't believe that the model did this week. It is obviously not overfitting because of the very high testing and validation scores which clearly show's the power of Yann LeCun!!

