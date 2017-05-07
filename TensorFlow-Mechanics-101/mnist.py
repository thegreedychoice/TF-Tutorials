"""
This file basically builds the model and the tensor graph of a mnist model with a feed-forward neural network
and ops to support the training and evaluation of the model.

"""


import tensorflow as tf
import numpy as np 	
import math

IMAGE_HEIGHT = 28
IMAGE_WIDTH = 28
NUM_IMAGE_PIXELS = IMAGE_HEIGHT * IMAGE_WIDTH
NUM_CLASSES = 10 #0-9


def inference(images, num_hidden1_nodes, num_hidden2_nodes):
	"""
	This methods builds a graph of feed-forward network which can basically infer or output a tensor representing a 
	prediction for a given input. 
	This inference graph is built based to place_holder inputs given to it in the parameters.
	This feed-forward network consists of two hidden layers with ReLu activations and a softmax output layer
	"""
	#this creates a scope for hidden layer 1 with its own paramters. 
	#the unique name given to this weights variable is 'hidden_layer1/weights'
	with tf.name_scope('hidden_layer1'):
		weights = tf.Variable(
			tf.truncated_normal([NUM_IMAGE_PIXELS, num_hidden1_nodes], stddev=1.0 / math.sqrt(float(NUM_IMAGE_PIXELS))),
			 name='weights') 
		biases = tf.Variable(tf.zeros([num_hidden1_nodes]), name='biases')
		#output of the first hidden layer
		o_hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)

	#this creates a scope for hidden layer 2 with its own paramters. 
	with tf.name_scope('hidden_layer2'):
		weights = tf.Variable(
			tf.truncated_normal([num_hidden1_nodes, num_hidden2_nodes], stddev=1.0/math.sqrt(float(num_hidden1_nodes))),
			name='weights')  
		biases = tf.Variable(tf.zeros([num_hidden2_nodes]), name='biases')
		#output of the second hidden layer
		o_hidden2 = tf.nn.relu(tf.matmul(o_hidden1, weights) + biases)

	#this creates a scope for output layer with its own paramters. 
	with tf.name_scope('output_softmax_layer'):
		weights = tf.Variable(
			tf.truncated_normal([num_hidden2_nodes, NUM_CLASSES], stddev=1.0/math.sqrt(float(num_hidden2_nodes))), name='weights')  
		biases = tf.Variable(tf.zeros([NUM_CLASSES]), name='biases')
		#final prediction 
		logits = tf.matmul(o_hidden1, weights) + biases

	return logits


def loss(logits, labels):
	"""
	This method adds additional ops to the graph build in inference(). Adding this allows us to estimate our parameters by 
	minimizing the total loss. We use cross entropy loss in this example.
	"""
	labels = tf.to_int64(labels)
	loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits), name='entropy_loss')
	return loss

def training(loss, eta):
	""" This method futher adds ops to the graph to minimize the loss by optimization using gradient descent.
	"""
	#this op basically generates summary values and emits snapshot value of loss everytime the summary is written out
	tf.summary.scalar('loss', loss)
	#for optimiziation we use a gradient descent optimizer with some learning rate, eta
	optimizer = tf.train.GradientDescentOptimizer(eta) 
	#this variable holds a counter with an initial value 0, to be used as a counter in training 
	global_counter = tf.Variable(0, name='global_counter', trainable=False)
	#this step is responsible for minimizing loss by calculating gradients and updating them
	#also attach the global counter to the train_op step to keep count of steps
	train_op = optimizer.minimize(loss, global_step=global_counter)
	return train_op

def evaluation(logits, labels):
	"""
	This method adds yet another op to the graph which basically evaluates the model by checking the count of 
	correct predictions by the model. We use a the tensor flow in_top_k method to check if our prediction is 
	in top k = 1
	"""
	#correct_preds is a boolean tensor of size [batch_size] holding True for items 
	#where lables which are top k=1 of all logits for that example
	correct_preds = tf.nn.in_top_k(logits, labels, k=1)
	#accuracy holds the sum value of correct predictions in the batch by first casting boolean to int and then
	#calculating the average mean over all the items
	correct_sum = tf.reduce_sum(tf.cast(correct_preds, tf.int32))
	return correct_sum







