# Neural Networks

[//]: # (Image References)
[perceptron]: ./Images_Writeup/perceptron.png "Perceptron"
[perceptron2]: ./Images_Writeup/perceptron2.png "Perceptron2"
[step-sigmoid]: ./Images_Writeup/step-sigmoid.png "step-sigmoid"
[discrete-continuous]: ./Images_Writeup/discrete-continuous.png "discrete-continuous"
[perceptron-step-sigmoid]: ./Images_Writeup/perceptron-step-sigmoid.png "perceptron-step-sigmoid"
[maximum-likelihood]: ./Images_Writeup/maximum-likelihood.png "maximum-likelihood"
[cross-entropy]: ./Images_Writeup/cross-entropy.png "cross-entropy"
[softmax]: ./Images_Writeup/softmax.png "softmax"


In this project, we'll train and build a deep neural network to drive a car in a simulator.

"Deep learning is just another term for using neural networks to solve problems".

"A neural network is a machine learning algorithm that you can train using an input (like camera images or sensor reading) and generate outputs (like the steering angle a car can take, or how fast it should go)".

"Classification problems are important for self-driving cars. Self-driving cars might need to classify whether an object crossing the road is a car, pedestrian, and a bicycle. Or they might need to identify which type of traffic sign is coming up, or what a stop light is indicating".

## Perceptron
Perceptron can be seen as a combination of nodes, where the first node calculates a linear equation on the inputs on the weights, and the second node applies the step function to the result

![perceptron]


### Weights
"How does it know whether grades or test scores are more important in making this acceptance decision?" Well, when we initialize a neural network, we don't know what information will be most important in making a decision. It's up to the neural network to learn for itself which data is most important and adjust how it considers that data.

It does this with something called *weights*.

When input comes into a perceptron, it gets multiplied by a weight value that is assigned to this particular input.
 
These weights start out as random values, and as the neural network learns more about what kind of input data leads to a student being accepted into a university for example, the network adjusts the weights based on any errors in categorization that results from the previous weights. This is called *training* the neural network. A higher weight means the neural network considers that input more important than other inputs, and lower weight means that the data is considered less important.

### Activation Function
The result of the perceptron's summation is turned into an output signal is done by feeding the linear combination into an activation function. 

Activation functions are functions that decide, given the inputs into the node, what should be the node's output? Because it's the activation function that decides the actual output, we often refer to the outputs of a layer as its "activation".

### Perceptrons as Logical Operators
Some logical operators can be represented as perceptrons, 

## Gradient Descent


A prediction is basically the answer we get from the algorithm.
A **discrete** answer is given in the form of a "yes/no".
A **continuous** answer is given in the form of a number - normally between 0 and 1 - considered a probability.

![discrete-continuous]


When working with gradient descent, we most likely want to work with **continous predictions** and, the way we move a predicton from beeing discrete to a continous one is by changing also the activation function, from the *Step Function* to the *Sigmoid Function*. 
![step-sigmoid]

Putting in a Perceptron way, these predictions can be shown like:
![perceptron-step-sigmoid]

In the examples above, it is shown how to predict two different variables, whether if it is red or blue. But, and how about the problems where we have 3 or more classes?

<mark>**Ler sobre Sigmoid Activation Function!!!11!Onze!**</mark>

## How to evaluate a model?
If we want to evaluate (and even improve) our model, one usual method used is the **Maximum Likelihood** method, which basically pick a model that gives the existing labels the highest probability. For example: Which one of the following examples is the model that better classified the red/blue dots (where the red labels are the probability of the dot to be red and the blue labels are the probability of the dot to be blue)? The model in the right is better, because it made the arrangement of the points much more likely to have those colors.

![maximum-likelihood]

In this example we only had 4 data points, so the product of the probabilities became a "readable" result. But what if we had thousands of data points? The product would be a number really low (like 0.000000002, for example). That's why the <mark>**cross-entropy**</mark> is generally used, which is value given by calculating the sum of the negative natural logarithm of the probabilities (it is negative because the logarithm of a number between 0 and 1 is always a negative number, so we invert it in order to get a positive number).

![cross-entropy]

It is important to notice that a <mark>good model gives a low cross-entropy, while a bad model gives a high cross-entropy </mark>.  

Read about backpropagation: https://karpathy.medium.com/yes-you-should-understand-backprop-e2f06eab496b#.vt3ax2kg9

https://www.youtube.com/watch?v=59Hbtz7XgjM


# Deep Learning
**Classification** is the task of taking an input and give it a label. Classification, or marginally prediction, is the central building block of machine learning.

In order to classify the data, it is needed to transform a score into a probability. To do so, we use a function called **SOFTMAX**. In tensorflow, it is applied by using:
<mark>tf.nn.softmax()</mark>.

![softmax]

<mark>LER SOBRE STOCHASTIC GRADIENT DESCENT!!</mark>

Gradient descent algorithms multiply the gradient by a scalar known as the learning rate (also sometimes called step size) to determine the next point. For example, if the gradient magnitude is 2.5 and the learning rate is 0.01, then the gradient descent algorithm will pick the next point 0.025 away from the previous point.

*"In deep learning, when things don't work, always try to lower the **learning rate** first."*

## Mini-batching
Mini-batching is a technique for training on subsets of the dataset instead of all the data at one time. This provides the ability to train a model, even if a computer lacks the memory to store the entire dataset.

The idea is to randomly shuffle the data at the start of each epoch, then create the mini-batches. For each mini-batch, you train the network weights with gradient descent. Since these batches are random, you're performing SGD with each batch. So, in order to run large models on your machine, you'll learn how to use mini-batching.

## Epochs
An epoch is a single forward and backward pass of the whole dataset. This is used to increase the accuracy of the model without requiring more data. 






---
additional resources about CNNs:

Andrej Karpathy's CS231n Stanford course on Convolutional Neural Networks.
https://cs231n.github.io/

Michael Nielsen's free book on Deep Learning.
http://neuralnetworksanddeeplearning.com/

Goodfellow, Bengio, and Courville's more advanced free book on Deep Learning.
https://www.deeplearningbook.org/