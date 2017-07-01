"""
This program demonstrates how to use batch normalization to test the model.
We would be using population mean and variance to test the model instead of batch mean and variance.
But we would calculate the population mean and variance during the training phase using the exponential moving average technique and implemet
these values later on.

It is a simpler version of Tensor flow's official version of batch normalization technique.
"""

def batch_norm_wrapper(inputs, is_training, decay=0.999, scope='batch_normalization'):
	with tf.variable_scope(scope):
		scale = tf.get_variable(tf.ones[inputs.get_shape()[-1]])