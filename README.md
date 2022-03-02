# **Traffic Sign Recognition** 

## Writeup

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

[image1]: ./examples/traffic_sign.png "Traffic_Sign"
[image2]: ./examples/visualization.png "Visualization"
[image3]: ./examples/visualization2.png "Visualization2"
[image4]: ./examples/rbg2gray.png "Grayscale"
[image5]: ./examples/augmented.png "Augmented"
[image6]: ./examples/learning_rate.png "Learning_Rate"
[image7]: ./examples/dropout.png "Dropout"
[image8]: ./examples/accuracy.png "Accuracy"
[image9]: ./examples/loss.png "Loss"
[image10]: ./examples/web_images.png "Web Images"
[image11]: ./examples/web_images_preprocessed.png "Preprocessed Web Images"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/LucasMirachi/CarND-Traffic-Sign-Classifier/blob/main/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic signs data set, which are:

* The size of training set is ?

*The training set is composed by **34799** traffic sign images*

* The size of the validation set is ?

*The validation set is composed by **4410**traffic sign images*

* The size of test set is ?

*The test set is composed by **12630** traffic sign images*

* The shape of a traffic sign image is ?

Each of the traffic sign images has a size of 32x32 pixels and 3 color channels (R,G,B).

![Traffic_Sign][image1]

* The number of unique classes/labels in the data set is ?

	The dataset has a total of **43 unique classes/labels**, according to the following extracted list:

      ClassId                                         SignName
      --------------------------------------------------------
      0                                   Speed limit (20km/h)
      1                                   Speed limit (30km/h)
      2                                   Speed limit (50km/h)
      3                                   Speed limit (60km/h)
      4                                   Speed limit (70km/h)
      5                                   Speed limit (80km/h)
      6                            End of speed limit (80km/h)
      7                                  Speed limit (100km/h)
      8                                  Speed limit (120km/h)
      9                                             No passing
      10           No passing for vehicles over 3.5 metric tons
      11                  Right-of-way at the next intersection
      12                                          Priority road
      13                                                  Yield
      14                                                   Stop
      15                                            No vehicles
      16               Vehicles over 3.5 metric tons prohibited
      17                                               No entry
      18                                        General caution
      19                            Dangerous curve to the left
      20                           Dangerous curve to the right
      21                                           Double curve
      22                                             Bumpy road
      23                                          Slippery road
      24                              Road narrows on the right
      25                                              Road work
      26                                        Traffic signals
      7                                            Pedestrians
      28                                      Children crossing
      29                                      Bicycles crossing
      30                                     Beware of ice/snow
      31                                  Wild animals crossing
      32                    End of all speed and passing limits
      33                                       Turn right ahead
      34                                        Turn left ahead
      35                                             Ahead only
      36                                   Go straight or right
      37                                    Go straight or left
      38                                             Keep right
      39                                              Keep left
      40                                   Roundabout mandatory
      41                                      End of no passing
      42      End of no passing by vehicles over 3.5 metric ...



#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the train data set using the *seaborn* library.

![Visualization][image2]

Looking at the graph, it is possible to observe that our data set doesn't has an uniform quantity of images per class - it actually has over 2000 images of some specific classes (like classes 1 and 2 - signs of Speed limit 30km/h 50km/h) and less than 250 images of other classes (like class 0 - signs of Speed limit of 20km/h). 

It is not very clear why this discrepancy exists... my guess is that perhaps the traffic signs with the most images in the data set represents the ones which are the most common signs to find on the roads in Germany? This would be a fun research to do in a future.

Also, it is interesting to visualize if, despite having different number of images for each class, the partitioning of training, validation and test subsets have the same proportions.

![Visualization2][image3]

Apparently, the three sets do have the same proportions for each class. It is important to notice that there is not a rule-of-thumb for how to divide the data set into training, validation and testing sets, but the most common ratios used are:

   *  70% train, 15% val, 15% test
   *  80% train, 10% val, 10% test
   *  60% train, 20% val, 20% test

Then, so far we can say that our data set is all good!

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. 

As first step, I decided to convert the images to grayscale mainly because grayscale simplifies the algorithm and reduces computational requirements. According to [LeCun's paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) we can see that, for the classification task, the traffic sign's color is not that relevant to increase the model's final accuracy.

Here is an example of a traffic sign image before and after grayscaling.

![Grayscale][image4]

According to Depeursinge's [Fundamentals of Texture Processing for Biomedical Image Analysis](https://www.sciencedirect.com/topics/engineering/image-normalization), image normalization ensures optimal comparisons across data acquisition methods and texture instances. The normalization of pixel values (intensity) is recommended for imaging modalities that do not correspond to absolute physical quantities. Keeping that in mind, to complete the prepossessing, I also normalized all images by dividing them by 255 so the pixel values of all images has a value between 0 and 1, making our model converge faster.

In addition to gray-scaling and normalizing the images, I tried to equalize them using the function cv2.equalizeHist in order to improve the contrast of our images, but the model didn't perform any better with it, so I decided to discard this processioning technique so our model can stay faster. 

Also, still according to [Sermanet and LeCun](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf), adding transformed images to our original training dataset synthetically will yield more robust learning to potential deformations in the test set. So, I experimented perturbing the samples in position ([-2,2] pixels), in scale ([.9,1.1] ratio) and rotation ([-15,+15] degrees) using keras.preprocessing.image library.

```
datagen = ImageDataGenerator(width_shift_range = 0.2,
                             height_shift_range = 0.2,
                             zoom_range = 0.1,
                             shear_range = 0.1,
                             rotation_range = 15)
datagen.fit(X_train)
```

Our final training image dataset became like the following images:

![Augmented][image5]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   							| 
| Convolution     	| 1x1 stride, padding = 'VALID', outputs 28x28x6 	|
| Activation				|		relu										|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution	    | 1x1 stride, padding = 'VALID', outputs 10x10x16      									|
| Activation		| relu        									|
| Max pooling				| 2x2 stride, outputs 5x5x16        									|
|	Flatten					|		Outputs 400										|
|			Fully Connected			|			Outputs 120									|
 |			Activation		|			relu									|
|			Dropout Layer			|		Keep Prob = 0.6								|
|			Fully Connected			|			Outputs 84									|
|			Activation		|			relu									|
|			Dropout Layer			|			Keep Prob = 0.6									|
|			Fully Connected			|			Outputs 43								|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used:

* 150 Epochs;
* Batch size of 128;
* Learning rate of 0.0009;
* Adam Optimizer;

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen? What were some problems with the initial architecture? How was the architecture adjusted and why was it adjusted?  Which parameters were tuned? How were they adjusted and why?

Initially, the chosen architecture was the LeNet-5 CNN using 100 Epochs, batch size of 128, learning rate of 0.01 and Adam Optimizer. However, the validation accuracy became way lower than our target of at least 93%. Here is the *"train of thought"* I took in order to adjust my architecture:

* **Lowered the learning rate**. Observing the generated loss graph, it was possible to see that at some moment, the loss value was not converging - it was bouncing from one high value to another, just like the right-most graph in the illustration bellow (took from this [Kagle's Learning Rate Article](https://www.kaggle.com/residentmario/tuning-your-learning-rate). So, to achieve an optimal learning rate, I tried using lr=0.0009 and it started converging in a much better rate!

 ![Learning_rate][image6]


* **Increased epochs**. Initially I was using epochs of 100, however, I could notice that the model loss and accuracy values were still decreasing when the 100th epoch was achieved. So I increased it to 150 epochs so the model could have more "time" to train and achieve more constant loss and accuracy values by the end of the training.

* **Added dropout layers to prevent overfitting**. After analyzing the first model accuracy plots, I observed that, once the model was achieving a validation accuracy lower than the training accuracy, we could consider that the model was overfitting. Among the main regularization techniques used to prevent overfitting in neural networks, I went for adding two dropout layers in the architecture. According to  Abhinav Sagar in *"Neural Networks, Overfitting"*, adding a dropout layer in an architecture will randomly drop neurons from the neural network during training in each iteration. When we drop different sets of neurons, it’s equivalent to training different neural networks. The different networks will overfit in different ways, so the net effect of dropout will be to reduce overfitting.

 ![Dropout - Adapted from Srivastava, Nitish, et al. ”Dropout: a simple way to prevent neural networks from
overfitting”, JMLR 2014][image7] 
 
Finally, here are the final loss and accuracy results (registered using the sess.run(loss_operation) and sess.run(accuracy_operation):

![Accuracy][image8] 
![Loss][image9] 

Test Loss = 0.2850, Test Accuracy = 94.173

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![Web Images][image10] 

The first image might be difficult to classify because of its displacement, once the sign is not plain as the ones in the dataset. Also, the model may have some trouble when classifying the second and fourth images, once they have a watermark on them, which can be confusing for the model identify it or not.

![Prepocessed Web Images][image11]

When preprocessing the images (which consists of resizing them to 32x32 pixels, converting to grayscale and normalizing them), all the images obviously lost some resolution and details. Also, the third and fifth images seems to have a lot of noisy pixels, which can also confuse our models prediction. 

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (50km/h)      		| Speed limit (50km/h)   									| 
| Yield     			| Yield 										|
| Right-of-way at the next intersection					| Right-of-way at the next intersection											|
| Go straight or right	      		| Go straight or right					 				|
| Speed limit (30km/h)		| Speed limit (30km/h)      							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%!! This compares favorably to the accuracy on the test set of 94.173%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. 


For the first image, the model is relatively sure that this is a stop sign (probability of 0.78), and the image does contain a 50km/h speed limit sign. The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.7756        			| Speed limit (50km/h)   									| 
| 0.2107     				| Double curve 										|
| 0.01378				| Speed limit (30km/h)											|
| 2.02321e-05	      			| Wild animals crossing					 				|
| 3.81858e-06				    | Road work     							|


For the second image the model was 100% sure it was a Yield sign.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0       			| Yield 									| 
| 0.0     				| Speed limit (20km/h) 										|
| 0.0				| Speed limit (30km/h)										|
| 0.0	      			| Speed limit (50km/h)				 				|
| 0.0				    | Speed limit (60km/h) 							|

For the third image as well the model was very close to classifying 100% correctly the traffic sign.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0       			| Right-of-way at the next intersection 									| 
| 8.81395e-12     				| Double curve 										|
| 2.17037e-13				| Beware of ice/snow										|
| 3.21533e-15 	      			|Priority road				 				|
| 1.18125e-15 				    | Pedestrians 							|

For the fourth image, the model was not so flawless, but it could correctly classify the traffic sign with a probability of over 80%.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.830474       			| Go straight or right									| 
| 0.0942405     				| End of all speed and passing limits 										|
| 0.064576				| Traffic signals			|
| 0.0106805 	      			|Speed limit (20km/h)			 				|
| 5.60259e-06 				    | Speed limit (30km/h) 							|
 
Finally, for the last image, the model did another prediction with a probability of almost 100%.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.999999       			| Speed limit (30km/h)									| 
| 1.42727e-06     				| Speed limit (50km/h)										|
| 2.8886e-09				| Speed limit (80km/h)			|
| 1.53915e-11  	      			|Speed limit (70km/h)			 				|
| 1.45847e-11 				    | Speed limit (20km/h) 							|


