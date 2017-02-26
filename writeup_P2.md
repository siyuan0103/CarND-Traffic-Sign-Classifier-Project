#**Traffic Sign Recognition** 

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

[image0]: ./count_train "count_train"
[image1]: ./count_test "count_test"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./extra-signs/1.jpg "Traffic Sign 1"
[image5]: ./extra-signs/2.jpg "Traffic Sign 2"
[image6]: ./extra-signs/3.jpg "Traffic Sign 3"
[image7]: ./extra-signs/4.jpg  "Traffic Sign 4"
[image8]: ./extra-signs/5.jpg  "Traffic Sign 5"
[image9]: ./overexp0 "overexp0"
[image10]: ./overexp1 "overexp1"
[image11]: ./overexp2 "overexp2"
[image12]: ./overexp3 "overexp3"
[image13]: ./overexp4 "overexp4"
[image14]: ./underexp0 "underexp0"
[image15]: ./underexp1 "underexp1"
[image16]: ./underexp2 "underexp2"
[image17]: ./underexp3 "underexp3"
[image18]: ./underexp4 "underexp4"
[image19]: ./sign1.png "sign1"
[image20]: ./sign2.png "sign2"
[image21]: ./sign3.png "sign3"
[image22]: ./sign_color.png "signcolor"
[image23]: ./sign_gray.png "signgray"
[image24]: ./sign_clahe.png "signclahe"
[image25]: ./sign1_color.png "sign1color"
[image26]: ./sign1_gray.png "sign1gray"
[image27]: ./sign1_equ.png "sign1equ"
[image28]: ./sign1_clahe.png "sign1clahe"
[image29]: ./sign0_color.png "sign0color"
[image30]: ./sign0_gray.png "sign0gray"
[image31]: ./extend0.png "extend0"
[image32]: ./extend1.png "extend1"
[image33]: ./extend2.png "extend2"
[image34]: ./extend3.png "extend3"
[image35]: ./extend4.png "extend4"
[image36]: ./extend5.png "extend5"
[image37]: ./extend6.png "extend6"
[image38]: ./extend7.png "extend7"
[image39]: ./extend8.png "extend8"
[image40]: ./count_train_extend "count_train_extend"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/siyuan0103/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 39209
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the IPython notebook.  

In this step the dataset are visualized in 3 ways:

* Randomly plot an image of each sign
* Plot the count of each sign in training and testing set
* Randomly plot 5 over-/underexposed images

Here are some of the traffic sign images as examples. We can find that the sign are all located at the center of the images. So the normalization of each pixel seems to be useful here. All of the are color images, but only with the difference of shape it's enough for us to tell every sign from another.

![alt text][image19] ![alt text][image20] ![alt text][image21]

Since there're 43 signs, not all the sign will be presented here. You can find them in the IPython notebook.

Then here is another exploratory visualization of the data set. It is a bar chart showing the distribution of the data in training set. Even though the distribution is significant non-uniform, the rarest sign has about 200 samples in the training set. It's not enough for the well training of a model. 

![alt text][image0]

In the testing set, the traffic signs have less samples but the same distribution as in the training set. The training and testing sets are proved to be evenly splitted. It's helpful for a correct testing accuracy.

![alt text][image1]

At last some over-/underexposed images would be randomly choosen and plotted. Those images, especially uderexposed images, are very hard to be recognized by human being. The histogram equalization seems to be helpful to handle those images.

![alt text][image11]![alt text][image12]![alt text][image13]
![alt text][image14] ![alt text][image15] ![alt text][image17]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the fourth code cell of the IPython notebook.

As a first step, I decided to convert the images to grayscale because the signs can be clearly distiguished by only the form. Though the color provides additional information for the recognization, but not necessary. In many case, especially with lower light conditions, color of images are distortional. So the color information are not reliable. By convert the images to grayscale, number of feature decreases by 2/3. It helps to prevent overfitting and can significantly speed up the training process. A further advantage from the grayscaling is that, the underexposed image can be easilier recongnized, just as the following example of a traffic sign image before and after grayscaling.

![alt text][image29]![alt text][image30]

After grayscaling, I try to increase the contrast by using contrast limited adaptive histogram equalization (CLAHE) function ```cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3,3))``` from opencv. The effect of this is significant, when the signs are not clear because of the overexposed background, as the following example shows (from left to right: color, graysclae, CLAHE). 

![alt text][image22]![alt text][image23]![alt text][image24]

Why not the more commonly used global histogram equalization? The global histogram equalization can also average the distribution of brightness, but when the background is underexposed, it also makes the sign too bright. The adaptive histogram equalization can overcome the dilemma, since the brightness are locally equalized, so more details are reserved. The following examples shows the advantage of adaptive histogram equalization (from left to right: color, grayscale, global histogram equalization, CLAHE).

![alt text][image25]![alt text][image26]![alt text][image27]![alt text][image28]

You can find more about the difference between global and local(adaptive) histogram equalization under  [this site](http://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html)

As a last step, I normalized the pixels (features) of the converted images by substracting the mean value and deviding the standard deviation of each pixel in training set as the recommended by [Standford cs231n Course](http://cs231n.github.io/neural-networks-2/) as the preprocessing for convolutional neural networks. I'm not sure about the principle, but it really works, by improving the testing accuracy from 8X% to 9X%. 

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

After the preprocessing of the data I generate some addtional data for training, because the distribution of each sign is not average. Some signs are well trained while some not. So I decide to add some transformation from the original images, and the number of extending images depends on the count of its sign.

Several transformation are commonly used in computer vision: reflection, translation, brightness changing, scaling and rotation. In the visualization of the images we can find that no reflection is needed here. And the sign are approximately placed in the center of the image. Since convolutional neural networks use a small filter to "scan" the 2D-image, it has tolerance for slightly offset. And the brightness has just been modified in the preprocesssing, so I decide to implement scaling and rotation here.

For  the sign appear more than 1600 times, no transformation will be implemented, since they are much enough.
For the sign with a count between 800 and 1600, they will be scaled by 1.3 time and 0.8 time.
For the sign with a count between 400 and 800, they will be seperately scaled by 1.3/0.8 time and rotated by +10/-10 degrees.
For the sign with a count less than 400, totally 8 combinations of rotation and scaling will be implemented.

Following are all possible transformations of an image. As the nomalized image cannot be plotted, the original image are grayscaled instead. (first row:original; second row:scaling; third row: rotation; fourth row:scaling with rotation):

![alt text][image31]
![alt text][image32]![alt text][image33]
![alt text][image34]![alt text][image35]
![alt text][image36]![alt text][image37]![alt text][image38]![alt text][image39]

Following bar charts present the difference between the extended data set (left) and the original data set (right). It's obvious that the shortage of samples with some signs are fixed. No sign has less than 1600 samples. The number of samples in training set (include validation set) increase from 39209 to 122365 by about 3 times.

![alt text][image40]![alt text][image0]

The code for splitting the data into training and validation sets is contained in the fifth code cell of the IPython notebook.  

To cross validate my model, I randomly split the training data into a training set and validation set. I did this by using the ```train_test_split``` function from ```sklearn.model_selection``` libraray.

My final training set had 110128 number of images. My validation set and test set had 12237 and 12630 number of images.


####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the seventh cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   							| 
| Convolution 5x5 	| 1x1 stride, valid padding, outputs 28x28x128 	|
| RELU					|												|
| Dropout		| keep probability 0.6   							|
| Max pooling	      	| 2x2 stride,  outputs 14x14x128 				|
| Convolution 5x5 	| 1x1 stride, valid padding, outputs 10x10x128 	|
| RELU					|												|
| Dropout		| keep probability 0.6   							|
| Max pooling	      	| 2x2 stride,  outputs 5x5x128 				|
| Flatten			| outputs 3200
| Fully connected		| outputs 200        							|
| Dropout		| keep probability 0.6   							|
| Fully connected		| outputs 100        									|
| Dropout		| keep probability 0.6   							|
| Fully connected		| outputs 43        									|
| Softmax				|         									|
 


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the seventh cell of the ipython notebook. 

To train the model, I used an Adamoptimizer. Learning rate 0.001, 0.0007, 0.0003, 0.0001, 0.00007 have been tested and 0.0003 are chosen because of the best balance of speed and accuracy. In the last lab it seems that smaller batch brings better accuracy, so here the batch size is set to be 32. Number of epochs is 20. More epochs bring no more significant improvement of testing accuracy.

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is also located in the seventh cell of the Ipython notebook.

My final model results were:
* validation set accuracy of 0.998 
* test set accuracy of 0.977

An modified LeNet approach was chosen here. At first I've tried LeNet because it has been proved to be suitable to handle the monochromatic images with simple contents. Based on it I've tried some variant, such as different size of weights in each layer, dropout, more convolution layers, etc, at last the current version brings the best test accuracy. The model tends to be overfitting so dropout is necessary and the keep probability of 0.5 to 0.9 were all tested and the model performances the best with keep probability of 0.6.


###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image6] ![alt text][image4] ![alt text][image8] 
![alt text][image5] ![alt text][image7]

The first image has big contrast between sign and background. With an unsuitable histogram equalization the sign might be underexposed. The second image is not hard, it's only a little overexposed. The fourth image has a sign with some shadow of twigs on it, which might lead to misunderstanding of the content. The third and fifith images have both stickers on the sign, which brings a big challenge for the robustness. The perspective change in the fifth image makes the round sign somehow like a diamond, especially with low resolution.

They were all downsized to 32x32 pixels and preprocessed the same as the above dataset.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Vehicles over 3.5 metric tons prohibited     		| Vehicles over 3.5 metric tons prohibited  									| 
| Speed limit (50km/h)    			| Speed limit (50km/h) 										|
| Ahead only					| Ahead only											|
| Road work      		| Road work				 				|
|Turn right ahead			| Turn right ahead 							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares similar to the accuracy on the test set of 97.7%.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is very certain that this is a Vehicles over 3.5 metric tons prohibited sign (probability of 0.995307), and the image does contain a Vehicles over 3.5 metric tons prohibited sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.995307        			| Vehicles over 3.5 metric tons prohibited   									| 
| 0.0046751			|  No passing										|
| 1.4576e-05					| End of no passing											|
| 1.72013e-06	      			| No passing for vehicles over 3.5 metric tons					 				|
| 8.61386e-07				    | End of all speed and passing limits      							|


For the second image, the model is also certain that this is a sign for Speed limit (50 km/h) (probability of 0.955611), just the same as the actual sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.955611        			| Speed limit (50km/h)   									| 
| 0.0162892			|  Speed limit (30km/h)									|
| 0.0129247					| Speed limit (80km/h)			 |
| 0.00938738	      			| Speed limit (60km/h)					 				|
| 0.00234053			    | Speed limit (70km/h)      							|

For the the third image, the model is extremly sure that this is a sign for Ahead only (probability of 0.0.999924) despite the sticker on the sign, which is also correct. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.999924        			| Ahead only  									| 
| 7.36085e-05			|  Turn left ahead									|
| 2.12672e-06					| Go straight or right				 |
| 2.79962e-07	      			| Yield					 				|
| 4.3748e-08			    | Speed limit (60km/h)      							|

For the the fourth image, the model is relatively certain that this is a sign for Road work (probability of 0.750464), which is still correct. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.750464        			| Road work 									| 
| 0.234022			|  Go straight or right									|
| 0.00446596					| Priority road				 |
| 0.00286453	      			| Children crossing					 				|
| 0.00211937			    | Dangerous curve to the right      							|

For the fifth image, the model is very certain that this is a sign for Turn right ahead (probability of 0.997445), and this time it's still right. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.997445        			| Turn right ahead   									| 
| 0.000608522			|  Yield										|
| 0.000588409					| Ahead only											|
| 0.000465771	      			| No passing for vehicles over 3.5 metric tons					 				|
| 0.000200141				    | Keep left      							|