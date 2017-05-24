import numpy as np 
import tensorflow as tf 


#Get the dataset from Tensorflow Library and do one hot encoding on the set
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#Build the computational graph and create an interactive session
#if not using Interactive Session, the entire computational graph should be built before launching graph
sess = tf.InteractiveSession()

#Intialize the placeholder values for training data
x = tf.placeholder(tf.float32, shape = [None, 784]) #Each mnist image is 28 by 28 pixel image
y = tf.placeholder(tf.float32, shape = [None, 10]) # Each label is represented one-hot encode format 10-D(zero through nine)

#Intialize the paramters 
W = tf.Variable(tf.zeros([784, 10])) #784 input features and 10 outputs
b = tf.Variable(tf.zeros([10]))		#10 Classes

#Intialize the Global Variables to zeros
sess.run(tf.global_variables_initializer())

MODEL_TYPE = 2   #1 for simple regression Mode, 2 for Deep Convolutional Networks

if MODEL_TYPE == 1 :
	print("Linear Regression Model")
	#Implement the Linear Regression Model
	o = tf.matmul(x, W) + b
	#Use Cross Entropy Loss Function
	#It applies the softmax on the model's unnormalized model prediction and sums across all classes, 
	#and tf.reduce_mean takes the average over these sums
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=o))
	#Create a Gradient Descent Optimizer with 0.5 learning rate
	optimizer = tf.train.GradientDescentOptimizer(0.5)
	#train_step will take care of minimizing loss by updating paramters after gradient calculations.
	#This needs to be called at every epoch of the training 
	train_step = optimizer.minimize(loss)
	#Train the Model by doing  gradient descent
	epochs = 1000
	batch_size = 100
	for _ in range(epochs):
		#for each iteration we get a sample of 100 training examples 
		batch = mnist.train.next_batch(batch_size)
		#We can use feef_dict to replace the tensors in computaional graph
		train_step.run(feed_dict={x: batch[0], y: batch[1]})

	#Evaluation of the Model
	#correct_prediction holds array of booleans [True, False, True,...] by comparing the true and predicted lablel for sample
	correct_predictions = tf.equal(tf.argmax(y,1), tf.argmax(o,1))
	#Convert the boolean list to float to get the accuracy mean/percentage
	accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
	print(accuracy.eval(feed_dict={x : mnist.test.images, y: mnist.test.labels}))

if MODEL_TYPE == 2:
	print("Convolution")
	def initialize_weights(shape):
		#Initialize weights with a normal distribution of standard deviation 0.1
		weights = tf.truncated_normal(shape, stddev=0.1)
		return tf.Variable(weights)

	def initialize_biases(shape):
		#Initialize with a positive intial bias to avoid dead neurons since we are using Rectified Linear Neurons
		biases = tf.constant(0.1, shape=shape)
		return tf.Variable(biases)

	#Define Convolution and Pooling Methods
	def conv_2d(input, filter, strides, padding, use_cudnn_on_gpu=None, data_format=None, name=None):
		"""
		Args:

		input: A Tensor. Must be one of the following types: half, float32, float64. A 4-D tensor. 
			   The dimension order is interpreted according to the value of data_format, see below for details.
		filter: A Tensor. Must have the same type as input. A 4-D tensor of shape [filter_height, filter_width, in_channels, out_channels]
		strides: A list of ints. 1-D tensor of length 4. The stride of the sliding window for each dimension of input. 
				 The dimension order is determined by the value of data_format, see below for details.
		padding: A string from: "SAME", "VALID". The type of padding algorithm to use.
		use_cudnn_on_gpu: An optional bool. Defaults to True.
		data_format: An optional string from: "NHWC", "NCHW". Defaults to "NHWC". 
					 Specify the data format of the input and output data. With the default format "NHWC", 
					 the data is stored in the order of: [batch, height, width, channels]. Alternatively, the format could be "NCHW", 
					 the data storage order of: [batch, channels, height, width].
		name: A name for the operation (optional).
		
		Returns:

		A Tensor. Has the same type as input. A 4-D tensor. The dimension order is determined by the value of data_format, 
				  see below for details.
		"""
		return tf.nn.conv2d(input, filter, strides, padding)

	def maxpool_2d(value, ksize, strides, padding, data_format='NHWC', name=None):
		"""
		Args:

		value: A 4-D Tensor with shape [batch, height, width, channels] and type tf.float32.
		ksize: A list of ints that has length >= 4. The size of the window for each dimension of the input tensor.
		strides: A list of ints that has length >= 4. The stride of the sliding window for each dimension of the input tensor.
		padding: A string, either 'VALID' or 'SAME'. The padding algorithm. See the comment here
		data_format: A string. 'NHWC' and 'NCHW' are supported.
		name: Optional name for the operation.
		
		Returns:

		A Tensor with type tf.float32. The max pooled output tensor.
		"""
		return tf.nn.max_pool(value, ksize, strides, padding)


		
	#Now create a deep network 

	#First Convolutional and Pooling Layer
	#The convolution will compute 32 features for each 5x5 patch. Create 32 output channels
	#Its weight tensor will have a shape of [5, 5, 1, 32]. 32 filters each of size (5,5,1) 
	#The first two dimensions are the filter height and width, the next is the number of input channels, 
	#We will also have a bias vector with a component for each output channel.
	filters_w_conv1 = initialize_weights([5,5,1,32])  #filters shape
	filters_b_conv1 = initialize_biases([32])

	#Before we apply the Conv layer, we need to reshape our input image into a 4-d tensor
	x_image = tf.reshape(x, [-1, 28, 28, 1]) #(batch_size, height, width, depth/color channels)

	net_conv1 = conv_2d(input=x_image, filter=filters_w_conv1, strides=[1, 1, 1, 1], padding='SAME') + filters_b_conv1
	o_conv1 = tf.nn.relu(net_conv1) #rectified linear activation function f = max(x, 0)
	o_pool1 =  maxpool_2d(value=o_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

	#The image size is now reduced to 14*14 after max pooling
	#Second Convolutional and Pooling Layer
	filters_w_conv2 = initialize_weights([5,5,32,64]) #64 filters of each shape (5,5,32)
	filters_b_conv2 = initialize_biases([64])
	net_conv2 = conv_2d(o_pool1, filters_w_conv2, [1, 1, 1, 1], 'SAME') + filters_b_conv2
	o_conv2 = tf.nn.relu(net_conv2) 
	o_pool2 = maxpool_2d(o_conv2, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

	#the image size has been reduced to 7x7 after 2nd Pooling Layer
	#Connect a fully-connected layer with 1024 hidden neurons to allow processing on the entire image. 
	#Reshape the output of shape (7,7,64) into a flattened array of vectors
	#Fully-Connected Layer 1
	hidden_neurons = 1024
	w_fcn1 = initialize_weights([7*7*64, hidden_neurons])
	b_fcn1 = initialize_biases([1024])

	inp_fcn1_flatten = tf.reshape(o_pool2, [-1, 7*7*64]) #Flatten the output from Pooling Layer
	net_fcn1 = tf.matmul(inp_fcn1_flatten, w_fcn1) + b_fcn1
	o_fcn1 = tf.nn.relu(net_fcn1)


	#Dropout Layer
	#To reduce Overfitting, we create this layer
	#There is probability assigned which represents the probability of keeping the 
	#output for a neuron during training.

	keep_prob = tf.placeholder(tf.float32)
	o_drop1 = tf.nn.dropout(o_fcn1, keep_prob)

	#Fully Connected/Readout Layer
	#This gives the final output layer of un-normalized probabilities for each label
	w_fc2 = initialize_weights([hidden_neurons,10]) #10 for class labels
	b_fc2 = initialize_biases([10])

	o = tf.matmul(o_drop1, w_fc2) + b_fc2

	#Training the Network

	#define Cross Entropy as Loss Function
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=o))
	#Use ADAM optimizer
	optimizer = tf.train.AdamOptimizer(1e-4)
	train_step = optimizer.minimize(loss)

	#Evaluate Predictions and Accuracy
	correct_predictions = tf.equal(tf.argmax(y,1), tf.argmax(o, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

	sess.run(tf.global_variables_initializer())

	#Optimization over epochs
	#Logging over over every 100th iteration to evaluate the model
	epochs = 20000
	for i in range(epochs):
		batch = mnist.train.next_batch(50)
		if i%100 == 0:
			train_accuracy = accuracy.eval(feed_dict={x: batch[0], y: batch[1], keep_prob: 1.0})
			print("Epoch No : %d   Training Accuracy : %g"%(i, train_accuracy))
		train_step.run(feed_dict={x: batch[0], y: batch[1], keep_prob: 0.5})

	#Test your Model With Test Data
	print("Testing the Model :")
	print("Test Accuracy : %g "%(accuracy.eval(feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0 })))












