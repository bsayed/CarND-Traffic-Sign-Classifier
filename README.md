#**Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images


[//]: # (Image References)

[training_examples]: ./examples/training_examples_per_image_class.png "training_examples_per_image_class"
[test_examples]: ./examples/test_examples_per_image_class.png "test_examples_per_image_class"
[preprocessing_example]: ./examples/preprocessing_example.png "preprocessing_example"
[transformation_of_a_single_image]: ./examples/transformation_of_a_single_image.png "transformation_of_a_single_image"
[six_images]: ./examples/six_images.png "six_images"
[validation_accuracy]: ./examples/validation_accuracy.png "validation_accuracy"

[sample1]: ./examples/sample1.png "sample1"
[sample2]: ./examples/sample2.png "sample2"
[sample3]: ./examples/sample3.png "sample3"
[sample4]: ./examples/sample4.png "sample4"
[sample5]: ./examples/sample5.png "sample5"
[sample6]: ./examples/sample6.png "sample6"


[image1]: ./sample_images/60_KPH_label_3.png "Traffic Sign 1"
[image2]: ./sample_images/no_entry_label_17.png "Traffic Sign 2"
[image3]: ./sample_images/stop_label_14.png "Traffic Sign 3"
[image4]: ./sample_images/straight_ahead_label_35.png "Traffic Sign 4"
[image5]: ./sample_images/turn_left_ahead_label_34.png "Traffic Sign 4"
[image6]: ./sample_images/yield_label_13.png "Traffic Sign 4"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration


The code for this step is contained in the fourth code cell of the IPython notebook.  

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799.
* The size of test set is 12630.
* The shape of a traffic sign image is (32, 32, 3).
* The number of unique classes/labels in the data set is 43.

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the fifth code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. It is a bar chart showing the number of training examples per 
image class. The second chart show the number of test example per image class.

![alt text][training_examples]

![alt_text][test_examples]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the "Pre-process the Data Set" section of the IPython notebook.

As a first step, I decided to convert the images to grayscale because I am using a modified version of the LeNet architecture
which has been used successfully with the MNIST dataset which actually comes in grayscale.

I augmented the dataset with the following set of transformations which were mentioned in the paper titled 
"Traffic Sign Recognition with Multi-Scale Convolutional Networks" by Sermanet et al. [1](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf)
1. Random rotation of the image with value between [-15,+15] degrees.
2. Random zoom with a value between [0.9,1.1].
3. Random shift with a value between [-3,3] in both the x-axis and y-axis direction.
4. Gaussian blur with value of one as I noticed that a value greater than one makes the image too blurry since
the image is actually small to begin with, just 32x32 pixel. The one here means that the blurring is based on the 
values of one neighbouring pixels.


The following figure illustrates the before mentioned set of transformations.

![alt text][preprocessing_example]

As a last step, I normalized and centered the values of the image data around zero with range -1 to 1
because, as mentioned in the CS231n course from Standford, it helps the gradient descent converge properly
compared to providing the NN with only positive or negative values. I used the following formula:
```
new_value = (old_value - 128.0) / 128.0
```

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The data set provided to use was already split into a training, validation and test sets of sizes 34799, 4410, and 12630 respectively.
To cross validate my model, I used the training set and validation set. Both the training and validation sets where augmented 
by generating additional data. For each image in the training and validation set I did the following:

1. Convert color image to grayscale.
2. Blur the grayscaled image.
3. Apply random rotation on the blurred image two times.
4. Apply random rotation on the grayscaled image two times.
5. Apply random scaling on the grayscaled image two times.
6. Apply random shifting on the grayscaled image in both x and y axis two times.

This results in 10 transformations in total for each image in the training and validation sets.

My final training set had 347990 number of images. My validation set had 44100 and the number of images in the test set did not change.

I decided not to augment the testing set to see the performance on the original test images without any transformations.
However, in the testing phase the same preprocessing is applied on the test set, grayscaling images and value normalization 
between -1 and 1.

Here is an example of the set of transformations done on the original image:

![alt text][transformation_of_a_single_image] 


####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the "Model Architecture" cell of the ipython notebook. 
The proposed model is a modified version of the LeNet Architecture that we worked on in class, it had good
performance on the MNIST data set hence, the choice to modify it mainly for the practical experience of doing so, 
rather than using off-the-shelf architectures like VGGNet or AlexNet which I think might yield better perform
better than my proposed architecture. The proposed model achieved 97% accuracy on the testing set. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   					| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 30x30x10 	|
| RELU					|												|
| Average pooling	    | 2x2 stride, Output = 15x15x10                 |
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x32   |
| RELU					|												|
| Max pooling	        | 2x2 stride, Output = 5x5x32                   |
| Fully connected		| Input = 800. Output = 400						|
| Dropout               | Probability 0.5                               |
| Fully connected		| Input = 400. Output = 120						|
| Dropout               | Probability 0.5                               |
| Fully connected		| Input = 120. Output = 84						|
| Output layer          | Input = 84. Output= Num of classes (43)       |
|						|												|
 


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the "Train, Validate and Test the Model" section of the ipython notebook. 

To train the proposed model I used the following parameters:

* The labels of the traffic signs has been converted to one-hot-encoding format and the softmax function
is used to calculate probabilities of the different classes.
* Used the tensorflow truncated_normal() function with mu= 0 and sigma =0.1 to make the input data have zero mean and 0.1 standard deviation.
* I used the cross entropy to calculate the loss rate (misclassification).
* Used the Adam optimizer with learning rate of 0.001 to minimize the loss rate. 
* The Adam Optimizer is a modified version of the Stochastic Gradient Descent optimizer.
* The model was trained for 20 epochs with batch size of 128.

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the "Train, Validate and Test the Model" section of the Ipython notebook.

My final model results were:
* validation set accuracy of illustrated in next figure:
![alt_text][validation_accuracy]

* test set accuracy of 97%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
I used the LeNet architecture since it had good performance with the MNIST data set.
* What were some problems with the initial architecture?
Its initial accuracy wasn't that good, between 84% and 87% on the validation set and was even lower on the test set, which was an indication of under fitting.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
I changed the filter size in both the first and second convolution layer to make them more deeper and I have added an extra fully connected layer to increase the overall network
learning capacity. The intuition behind that is that the traffic sign images has more lines, edges, and curves than the MNIST data set of Arabic Numerals.
In addition I changed the first pooling layer from max-pooling to average pooling so that we do not lose too much information between the first and second conv layers.
* Which parameters were tuned? How were they adjusted and why?
I experimented with changing the number of epochs, learning rate and batch size but settled on 20 epochs, 0.001 learning rate and 128 batch size. 
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
I used average pooling rather than max pooling as the max pooling tend to lose more information than averaging. For example,
if a 2x2 max-pooling layer is used 3 out of every 4 values are thrown away.

If a well known architecture was chosen:
* What architecture was chosen? AlexNet
* Why did you believe it would be relevant to the traffic sign application? Very good performance in [ImageNet ILSVRC competition](http://www.image-net.org/challenges/LSVRC/). 
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
The validation accuracy is around 94% and the test accuracy is about 95%. The test data has not been seen before by the network 
and no transformation has been done on it, just grayscaling and normalization of values to be between -1 and 1.
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are six German traffic signs that I found on the web:

![alt_text][six_images]

I found this image at the following [link.](http://forums.fast.ai/t/dataset-discussion-german-traffic-signs/766)

The first, forth, and fifth images might be difficult to classify because are taken from an angle. They are a little
bit tilted to the left. The second and third images might be difficult to classify because they are dark images.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 60 km/h	      		| 60 km/h					 				    |
| No Entry              | No Entry                                      |
| Stop Sign      		| Stop sign   									| 
| Ahead only            | Ahead only                                    |
| Turn left ahead       | Turn left ahead                               |
| Yield					| Yield											|


The model was able to correctly guess all the traffic signs, which gives an accuracy of 100%. 
This compares favorably to the accuracy on the test set of 97%

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the "Step 3: Test a Model on New Images" section of the Ipython notebook.
I was kind of surprised of the prediction accuracy since it was 100% to the point that I am suspecting that these test images
could be part of the training data to being with, however, I wasn't able to verify that but still the numbers makes sense to me.

For the 1st image: 

![alt_text][sample1]


For the 2nd image: 

![alt_text][sample2]


For the 3rd image: 

![alt_text][sample3]


For the 4th image: 

![alt_text][sample4]


For the 5th image: 

![alt_text][sample5]


For the 6th image: 

![alt_text][sample6]

