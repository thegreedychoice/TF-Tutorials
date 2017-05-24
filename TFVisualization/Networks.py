import tensorflow as tf
import numpy as np


def initialize_weights(shape, std=0.1):
    # Initialize weights with a normal distribution of standard deviation 0.1
    weights = tf.truncated_normal(shape, stddev=std)
    return tf.Variable(weights)


def initialize_biases(shape, bias=0.1):
    # Initialize with a positive intial bias to avoid dead neurons since we are using Rectified Linear Neurons
    biases = tf.constant(bias, shape=shape)
    return tf.Variable(biases)

def net_convolution2d(input_x, num_filter, filter_shape, strides, padding,
	 use_cudnn_on_gpu=None, data_format=None, name=None, weights_standard_dev=None, bias_constant=None):
	"""
	Performs Convolution and adds bias values as well
	:param
		input_x : The input dataset (no labels)
		num_filter : Number of kernels or filters
		filter_shape : Shape of a filter in tuple form (height, width, depth)
		strides : The stride of the sliding window for each dimension of the input of the convolution. 
				  e.g.[1,1,1,1]
		padding : A string from: "SAME", "VALID". The type of padding algorithm to use.
		use_cudnn_on_gpu: An optional bool. Defaults to True.
		data_format: An optional string from: "NHWC", "NCHW". Defaults to "NHWC". 
					 Specify the data format of the input and output data. 
					 With the default format "NHWC", the data is stored in the 
					 order of: [batch, input_height, input_width, input_channels]. 
					 Alternatively, the format could be "NCHW".
		name: A name for the operation (optional).
		weights_standard_dev : Standard Deviation for normal weights distribution.
		bias_constant :  Some positive value of bias
	:return
		A Tensor for the convolutional layer with all the desired specifications of type tf.float32.
	"""
	weights_standard_dev = 0.1
	bias_constant = 0.1
	shape_w = list(filter_shape)
	shape_w.append(num_filter)
	#Initialize weights with a normal distribution of standard deviation 0.1
	weights = initialize_weights(shape_w, weights_standard_dev)
	#Initialize with a positive intial bias to avoid dead neurons since 
	#we are using Rectified Linear Neurons
	shape_b = [num_filter]
	biases = initialize_biases(shape_b, bias_constant)

	return (tf.nn.conv2d(input=input_x, filter=weights, strides=strides, padding=padding,
							use_cudnn_on_gpu=use_cudnn_on_gpu, data_format=data_format, name=name)
	+ biases)
	
def max_pool2d(input, pool_size, strides, padding, data_format='NHWC', name=None):
    """
    :param
    	input: A 4-D Tensor with shape [batch, height, width, depth] and type tf.float32.
    	pool_size: A list of ints that has length >= 4. The size of the window for each dimension
    		   of the input tensor.
    	strides: A list of ints that has length >= 4.
    			 The stride of the sliding window for each dimension of the input tensor.
    	padding: A string, either 'VALID' or 'SAME'.
    	data_format: A string. 'NHWC' and 'NCHW' are supported.
    	name: Optional name for the operation.
    :return
    	A Tensor with type tf.float32. The max pooled output tensor.

    """
    return tf.nn.max_pool(value=input, ksize=pool_size, strides=strides, padding=padding)

def reLu(input, name=None):
    """

    :param
        input: A Tensor. Must be one of the following types:
                float32, float64, int32, int64, uint8, int16, int8, uint16, half
        name : Optional name of the operation.
    :return:
        A Tensor of same type as input.
    """
    return tf.nn.relu(features=input, name=name)

def dropout(input, keep_prob, noise_shape=None, seed=None, name=None):
    """
    :param
        input: The input to the Dropout layer
        keep_prob:  The probability of keeping the output for a neuron during training, i.e. the probability that
                    each element is kept.
        noise_shape: A 1-D Tensor of type int32, representing the shape for randomly generated keep/drop flags.
        seed: A Python integer. Used to create random seeds. See tf.set_random_seed for behavior.
        name: A name for this operation (optional).
    :return:
        A Tensor of same shape as input
    """
    return tf.nn.dropout(x=input, keep_prob=keep_prob, noise_shape=noise_shape, seed=seed, name=name)

