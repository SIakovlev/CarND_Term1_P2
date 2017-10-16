# **Traffic Sign Recognition** 

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

[image1]: ./report_images/histogram.jpg "Visualization"
[image2]: ./report_images/Normalisation.jpg "Normalisation"
[image3]: ./report_images/Augmentation.jpg "Augmentation"
[image4]: ./report_images/0_original.jpg "Traffic Sign 1"
[image5]: ./report_images/1_original.jpg "Traffic Sign 2"
[image6]: ./report_images/2_original.jpg "Traffic Sign 3"
[image7]: ./report_images/3_original.jpg "Traffic Sign 4"
[image8]: ./report_images/4_original.jpg "Traffic Sign 5"
[image9]: ./report_images/0.jpg "Traffic Sign 1 guess"
[image10]: ./report_images/1.jpg "Traffic Sign 2 guess"
[image11]: ./report_images/2.jpg "Traffic Sign 3 guess"
[image12]: ./report_images/3.jpg "Traffic Sign 4 guess"
[image13]: ./report_images/4.jpg "Traffic Sign 5 guess"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/SIakovlev/CarND_Term1_P2/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the `matplotlib` library to calculate summary statistics of the traffic
signs data set:

* The size of training set is `34799` RGB images
* The size of the validation set is `4410` RGB images
* The size of test set is `12630` RGB images
* The shape of a traffic sign image is `(32,32,3)`
* The number of unique classes/labels in the data set is `43` (I used `numpy.unique`: [link](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.unique.html))

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is spread across the classes

![alt text][image1]

As we can see, all datasets have similar distributions.

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

Data processing was done in two steps:

* Adaptive histogram equalisation [CLAHE](https://www.wikiwand.com/en/Adaptive_histogram_equalization). This method improves the image contrast by computing histograms for diffrent parts and redistributes lightness over the image. In the case of German Traffic Sign Database, many images are too dark or too bright, hence brightness balancing will potentially help to make datasets more uniform and improve classification accuracy.

The described procedure can be done in three steps (taken from [here](https://stackoverflow.com/questions/31998428/opencv-python-equalizehist-colored-image)):
    * Conversion to [YUV color space](https://en.wikipedia.org/wiki/YUV)
    * CLAHE algorithm is applied to Y channel
    * Conversion back to BGR color space

I used `opencv` library functions (in particular `cv2.createCLAHE` for histogram equlisation) and this [tutorial](https://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html).

* Normalisation step. I did it using the formula from [wiki article](https://en.wikipedia.org/wiki/Normalization_(image_processing)) for linear normalisation. 

Here is the example of appying these steps to a traffic sign image:

![Data processing][image2]

The German Traffic Sign Database is very nonuniform, i.e. it contains more images of one class than another. In addition the images belonging to one class can be very different from neural network point of view, i.e. the sign can be blurred, or shifted, or seen from different angles (which causes perspective distortion), etc. To model the data variations within a single class I decided to augment a dataset. Data augmentation was done in several ways:

* **rotation** by a random angle (between -20 and 20 degrees) - models basic uncertainty of the sign angle with respect to the picture frame.
* **translations** - models traffic signs at different positions on the picture
* **affine transformation** - models perspective distortion effect ()
* **brightness variation** - models variation of brightness level for original dataset
* their combinations (see function `image_augment(X)`)

Here is an example of an original image and an augmented image with methods listed above:

![alt text][image3]

The augmented dataset can be obtained in two ways: 
* Nonbalanced set augmentation. I randomly apply transformations listed above to each element of the original dataset and stack them together (function `set_augment`). The augmented dataset contains two times more images than the original one, i.e. its shape`(69598, 32, 32, 3)` and has the same distribution of sign images across the sign types. 
* Balanced set augmentation (`set_augment_balanced`), i.e. transformations applied to the images of each class in such a way that the augmented set has uniform distribution of images across the classes.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							      | 
| Convolution 1: 5x5x18    | 1x1 stride, valid padding, outputs 28x28x18 	|
| RELU					|												            |
| Max pooling	      | 2x2 stride,  outputs 14x14x18 				      |
| Convolution 2: 3x3x48	   | 1x1 stride, valid padding, outputs 12x12x48   |
| RELU					|												            |
| Max pooling	      | 2x2 stride,  outputs 6x6x48 				      |
| Convolution 3: 3x3x96	   | 1x1 stride, valid padding, outputs 4x4x96   |
| RELU					|												            |
| Max pooling	      | 2x2 stride,  outputs 2x2x96 				      |
| Fully connected		| Input: convolution layer outputs (5640), outputs: 688        					|
| RELU					|												            |
| Dropout            | Keep rate: 0.5                                |
| Fully connected		| Input: 688, outputs: 86        					|
| RELU					|												            |
| Dropout            | Keep rate: 0.5                                |
| Fully connected		| Input: 86, outputs: 43        					|
| Softmax				|         									|
 
#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

* Optimiser type: [Adam Optimiser]() - in [CS231n](http://cs231n.github.io/neural-networks-3/) Adam is suggested as a default algorithm to use for majority of applications. In the original [paper](https://arxiv.org/abs/1412.6980) it shows the best performance for MNIST dataset.
* Batch size: 128 - I left default value as it worked good for me. Number of epochs: 150 - after about 150 epochs I didn't see any noticeable improvements.

* For data augmentation I prepared 3 augmented datasets with the same distributions as the original one and 1 balanced dataset. During the training, at the start of each epoch we choose one dataset arbitrary. The functions `set_augment(X)` and `set_augment_balanced(X)` choose the method for image augmentation randomly as well. Therefore, since the augmented datasets are different, by randomly choosing them at the start of each epoch we avoid overfitting and keep the augmented part of training dataset constantly changing. The idea of blending nonbalanced dataset with a balanced one is borrowed from [here](https://navoshta.com/traffic-signs-classification/) and slightly simplified. As result, the described procedure forces a neural network to generalise. 

* Learning rate. The learning rate is reduced with the number of epochs:
    * 0 - 80 epochs: 0.001 
    * 80 - 120 epochs: 0.0001
    * 120 - 150 epochs: 0.00001
  
  The idea behind it is to switch to a lower learning rate when training process achieves plateau. By doing so it helps with improving training accuracy. The `Traffic_Sign_Classifier.ipynb` file contains training information at each epoch and it is noticeable that after lowering training rate, the accuracy on training set grows.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* **training set** accuracy of 99.9%
* **validation set** accuracy of 98.8% 
* **test set** accuracy of 97.8%
* Time: it took less than one hour to get these results on g2.2xlarge machine.

The architecture design choice:

* The first architecture I tried was LeNet that we used for MNIST dataset in the lab and the accuracy was about 89% on test dataset. There were a few problems I have noticed:
    * Overfitting, i.e. training accuracy was much higher than validation accuracy
    * Small network capacity, i.e. training accuracy does not achieve at least 99%. In other words this architecture does not allow to quickly learn all the details from given dataset.
* To avoid overfitting I added two dropout layers (see architecture description above). To resolve the second problem a one convolution layer was added and output of each convolution layer was connected to the first fully connected layer. The last architectural solution was borrowed from the [paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) by Pierre Sermanet and Yann LeCun. The new architecture provides with a good training rate and allows to achieve a very high accuracy on the training set.
* The number of epochs was increased to 100 so that training curve achieves plateau for both training and validation sets.
* **What are some of the important design choices and why were they chosen?** In summary there are 3 important design choices:
    * One extra convolution layer - increases network capacity
    * Two dropout layers - reduce overfitting
    * Outputs of each convolution layer is connected to a fully connected layer

* **How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?**
Based on the model results, training, validation and test accuracies are very close to each other and are about 98-99% each. This means that the network generalised dataset quite well. However it is important to keep in mind that the quality of neural network model heavily depends on the dataset: its diversity, size, distributions, etc. Therefore if the dataset in not representing the modelling objective good enough, then even good results for each set do not mean that the model is working well. The next section is demonstrating this.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web (resized, the original images are [here](https://github.com/SIakovlev/CarND_Term1_P2/tree/master/test_images)). Since during the trainig I mainly used unbalanced dataset (see previous section) it is reasonable to expect that my model is biased with respect to the most frequent images from training set of images. In this section I deliberately chose some pictures in such a way that network will be forced to choose the most frequent label, even though it is not correct.

* Right-of-way at the next intersection. This image should be easy to classify because even after formatting the sign is clearly distinguishable. Just a simple example where neural network should not make a wrong prediction:

![alt text][image4]

* Road work. Completely unclear what is on the image - just a bunch of pixels that can be interpreted in many ways. However the trainig dataset contains many images of this type, which should mean that neural network will guess it correctly:

![alt text][image5] 

* Stop. The sign here is srinked and after formatting the inscription "Stop" is distinguishable and can be interpreted by neural network as a white line (i.e. "No entry" sign) which is better represented in the training dataset. I also picked this picture in order to check robustness of the model to affine transformations:

![alt text][image6] 

* No entry. This picture has some weird white square. The images in dataset did not contain anything similar. However, this picture is well represented in the training set, that is why it should not be a problem. Also, the main sign is shifted up therefore it is a good way to check robustness of the model to tranlations:

![alt text][image7]

* Children crossing. It is completely unclear what is on the image. Again this picture can be interpreted in many ways. For instance it is similar to the "Bicycles crossing" sign which is better represented in the training set:

![alt text][image8]


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Right-of-way at the next intersection      		| Right-of-way at the next intersection   									| 
| Road work     			| Road work 										|
| Stop					| No entry											|
| No entry	      		| No entry					 				|
| Children crossing			| Bicycles crossing    							|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. This cannot be compared to the accuracy on the test set because some of the images were taken to check how biased my model is. It is noticeable that the model treated a stop sign with corrupted inscription as the "No entry" sign as well as "Children Crossing" as the "Bicycles crossing" sign just because they were seen more often by the network during the training process. The next section provides top 5 gueses, whereform the decision process of the network becomes much clearer.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

* For the first image, the model is absolutely sure that this is a "Right-of-way at the next intersection" sign, which is expectable. The top five soft max probabilities were:

![alt text][image9]

* For the second image, the model recognises it correctly as well, even though the image is very corrupted and can be interpreted in many other ways. The reason why the network identifies it right is because the training set contains many images of this class, i.e. the model is just biased with respect to this class:

![alt text][image10]

* For the third image, the model fails. Here it had two most probable options: stop sign and no entry sign. However during the training it saw images of "No entry" class more often than images with "Stop" sign, therefore it tends to choose "No entry", which is wrong in this case:

![alt text][image11]

* For the forth image, the model...:

![alt text][image12]

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


