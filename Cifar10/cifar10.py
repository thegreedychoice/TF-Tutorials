"""This file will contain methods to build an image classification model Cifar-10 which will later be
trained to perform classification on the Cifar-10 dataset

"""

import sys
import os

import tensorflow as tf
import numpy as np
import cifar10_input

#tf.app.flags is the global container for flags in TensorFlow
#we can add and access flags and their values from it
FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of images to process in one batch.""")
data_dir =  os.path.dirname(os.path.abspath(__file__))+'/tmp/cifar10_data'
tf.app.flags.DEFINE_string('data_dir', data_dir,
                           """Path to the CIFAR-10 data directory.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")

# Global constants describing the CIFAR-10 data set.
IMAGE_SIZE = cifar10_input.IMAGE_SIZE
NUM_CLASSES = cifar10_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL


# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'


def activation_summary(x):
	"""This method creates summaries for activations which can be later viewed on tensorboard"""
	#We first need to remove 'tower_[0-9]/' from names incase this is a multi-GPU training Op, for summary visualization
	op_name = re.sub('%s_[0-9]*/'%TOWER_NAME, '', x.op.name)
	#Next we need to print the summaries into the hidden files
	#First Record the histogram summary of the activations
	tf.summary.histogram(op_name+'/activations', x)
	#Record the summary of the sparsity of the activations
	tf.summary.scalar(op_name+'/sparsity', x)

def variable_on_cpu(name, shape, initializer):
	"""This method creates a varible to store on the CPU memory"""

	with tf.device('/cpu:0'):
		dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
		var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
	return var

def variable_with_weight_decay(name, shape, stddev, wd):

    """This method creates an initialized Variable with weight decay.
  	Note that the Variable is initialized with a truncated normal distribution.
  	A weight decay is added only if one is specified.

  	Args:
    	name: name of the variable
    	shape: list of ints
    	stddev: standard deviation of a truncated Gaussian
    	wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.
  	Returns:
    	Variable Tensor
  	"""
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = variable_on_cpu(name,
                          shape,
                          tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)

    return var

def distorted_inputs():
	"""Construct distorted input for CIFAR training using the Reader ops.

    Returns:
        images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
        labels: Labels. 1D tensor of [batch_size] size.
    Raises:
        ValueError: If no data_dir
    """


	if not FLAGS.data_dir:
		raise ValueError('Please supply a data_dir')
	  	data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
	  	images, labels = cifar10_input.distorted_inputs(data_dir=data_dir,
	                                                  batch_size=FLAGS.batch_size)
	if FLAGS.use_fp16:
	  	images = tf.cast(images, tf.float16)
		labels = tf.cast(labels, tf.float16)
  	return images, labels


def inputs(eval_data):
	"""Construct input for CIFAR evaluation using the Reader ops.

	  Args:
	    eval_data: bool, indicating if one should use the train or eval data set.
	  Returns:
	    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
	    labels: Labels. 1D tensor of [batch_size] size.
	  Raises:
	    ValueError: If no data_dir
	"""

	if not FLAGS.data_dir:
		raise ValueError('Please supply a data_dir')
	  	data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
	  	images, labels = cifar10_input.inputs(eval_data=eval_data,
	                                        data_dir=data_dir,
	                                        batch_size=FLAGS.batch_size)
	if FLAGS.use_fp16:
		images = tf.cast(images, tf.float16)
	   	labels = tf.cast(labels, tf.float16)

		return images, labels


def inference(input):
	"""This method builds the CIFAR-10 Model/network which gives predictions given a dataset of images training or evaluation
  	Args:
    	data : dataset of CIFAR-10 images
  	Returns:
    	Logits : classification prediction of the set of images in order
    """
	#We also would be initializing all our tensorflow variables using tf.get_variable() and not tf.Variable()
	# #because tf.get_variable() lets us share our variables across multiple gpu runs for training
	#and tf.Variable() is the way to go if we would like to train our model on a single GPU

	with tf.variable_scope('convlayer_1') as scope:
		filters = variable_with_weight_decay('weights',
											 shape=[5, 5, 3, 64],
											 stddev=5e-2,
											 wd=0.0)
		# Checkout this link to learn more about convolutional layer
		# https://www.tensorflow.org/api_docs/python/tf/nn/conv2d
		conv = tf.nn.conv2d(input, filters, [1, 1, 1, 1], padding='SAME')
		biases = variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
		pre_activation = tf.nn.bias_add(conv, biases)
		out_conv1 = tf.nn.relu(pre_activation, name=scope.name)
		activation_summary(out_conv1)

		# Pooling Layer 1
		# Shape of the ksize and Strides is default "NHWC"
		# NHWC [batch, height, width, channels]
		poollayer_1 = tf.nn.max_pool(out_conv1,
									 ksize=[1, 3, 3, 1],
									 strides=[1, 2, 2, 1],
									 padding='SAME',
									 name='poollayer_1')
		# We perform normalization of the data from pooling layer before sending to the next layer
		# This is in accordance with the ImageNet classification paper described in the link
		# http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks
		# Normalization is useful to prevent neurons from saturating when
		# inputs may have varying scale, and to aid generalization.

		normalized_1 = tf.nn.lrn(poollayer_1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
								 name='normalized_1')

	# Convolution Layer 2
	with tf.variable_scope('convlayer_2') as scope:
		filters = variable_with_weight_decay('weights',
											 shape=[5, 5, 64, 64],
											 stddev=5e-2,
											 wd=0.0)
		conv = tf.nn.conv2d(normalized_1, filters, [1, 1, 1, 1], padding='SAME')
		biases = variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
		pre_activation = tf.nn.bias_add(conv, biases)
		out_conv2 = tf.nn.relu(pre_activation, name=scope.name)
		activation_summary(out_conv2)

		# This time normalization is perform on the outout generated by convolutional layer
		normalized_2 = tf.nn.lrn(out_conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
								 name='normalized_2')
		# Pooling Layer 2
		poollayer_2 = tf.nn.max_pool(normalized_2,
									 ksize=[1, 3, 3, 1],
									 strides=[1, 2, 2, 1],
									 padding='SAME',
									 name='poollayer_2')





	# local3 is the first fully connected layer with 384 hidden neurons
	with tf.variable_scope('local3') as scope:
		# Move everything into depth so we can perform a single matrix multiply.
	    #-1 in tf.reshape is used to infer the required shape automatically by TensorFlow
	    #after allocating the first dimension of the size as batch_size
	    reshaped_input = tf.reshape(poollayer_2, [FLAGS.batch_size, -1])
	    #dim is dimension of the individual input image that would be sent to fully connected layer
	    dim = reshaped_input.get_shape()[1].value
	    weights = variable_with_weight_decay('weights',
	    									shape=[dim, 384],
	                                        stddev=0.04,
	                                        wd=0.004)
	    biases = variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
	    local3 = tf.nn.relu(tf.matmul(reshaped_input, weights) + biases, name=scope.name)
	    activation_summary(local3)

	# local4 is the second fully connected layer with 192 hidden neurons
	with tf.variable_scope('local4') as scope:
		weights = variable_with_weight_decay('weights', shape=[384, 192],
												 stddev=0.04, wd=0.004)
		biases = variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
		local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
		activation_summary(local4)


	# linear layer(WX + b),
	# We don't apply softmax here because
	# tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
	# and performs the softmax internally for efficiency.
	with tf.variable_scope('softmax_linear') as scope:
		weights = variable_with_weight_decay('weights', [192, NUM_CLASSES],
										 stddev=1 / 192.0, wd=0.0)
		biases = variable_on_cpu('biases', [NUM_CLASSES],
						 tf.constant_initializer(0.0))
		softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
		activation_summary(softmax_linear)

	return softmax_linear

def loss(logits, labels):
	"""
	This functions calculates the average cross entropy loss for the given set of input
	Add summary for "Loss" and "Loss/avg".
  	Args:
    	logits: Logits from inference().
    	labels: Labels from distorted_inputs or inputs(). 1-D tensor
        	    of shape [batch_size]
  	Returns:
    	Loss tensor of type float.

	"""

	#Convert the labels into integars
	labels = tf.cast(labels, tf.int64)
  	cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, 
  																logits=logits, 
  																name='cross_entropy_per_example')
  	cross_entropy_mean = tf.reduce_mean(cross_entropy_loss, name='cross_entropy')
  	#The loss is added to the collection for the purpose of building the graph
  	tf.add_to_collection('losses', cross_entropy_mean)

  	# The total loss is defined as the cross entropy loss plus all of the weight
  	# decay terms (L2 loss).
  	# Perform L2 Regularization on the losses
  	return tf.add_n(tf.get_collection('losses'), name='total_loss')


def add_loss_summaries(total_loss):
  """Add summaries for losses in CIFAR-10 model.
  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.
  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  #Checkout the following link on Moving Averages
  #https://www.tensorflow.org/versions/r0.11/api_docs/python/train/moving_averages
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.summary.scalar(l.op.name + ' (raw)', l)
    tf.summary.scalar(l.op.name, loss_averages.average(l))

  return loss_averages_op






