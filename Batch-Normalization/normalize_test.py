"""
This program demonstrates how to use batch normalization to test the model.
We would be using population mean and variance to test the model instead of batch mean and variance.
But we would calculate the population mean and variance during the training phase using the exponential moving average technique and implement
these values later on.

It is a simpler version of Tensorflow's official version of batch normalization technique.
"""

import numpy as np 
import tensorflow as tf, tqdm
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt 

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)



#Intialize small epsilon value for Batch Normalization transform 
epsilon = 1e-3

eta = 0.01



def reset_graph():
	"""
	This method resets the Tensorflow Session's Default Graph
	"""
	if 'sess' in globals() and sess:
		sess.close()
	tf.reset_default_graph()



def batch_norm_wrapper(inputs, is_training, decay=0.999, scope_name='batch_normalization'):

	scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
	beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
	#Make them non-trainable so that it doesn't get updated during back-prop by the optimizer
	#we will update them in our won way
	pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
	pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)


	if is_training:
		batch_mean, batch_var = tf.nn.moments(inputs, [0])
		train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
		train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))

		with tf.control_dependencies([train_mean, train_var]):
			return tf.nn.batch_normalization(inputs, 
					batch_mean, batch_var, beta, scale, epsilon)
	else:
		return tf.nn.batch_normalization(inputs,
			pop_mean, pop_var, beta, scale, epsilon)

def build_graph(is_training):

	#Generate predetermined random weights so that the networks are similarly initialized
	w_input_hidden = tf.constant(np.random.normal(size=(784,100)).astype(np.float32))
	w_hidden_hidden = tf.constant(np.random.normal(size=(100,100)).astype(np.float32))
	w_hidden_output = tf.constant(np.random.normal(size=(100, 10)).astype(np.float32))

	#Placeholders
	x = tf.placeholder(tf.float32, shape=[None, 784])
	y_ = tf.placeholder(tf.float32, shape=[None, 10])

	#Layer 1
	w1 = tf.Variable(w_input_hidden)
	z1 = tf.matmul(x, w1)
	bn1 = batch_norm_wrapper(z1, is_training)
	l1 = tf.nn.sigmoid(bn1)

	#Layer 2
	w2 = tf.Variable(w_hidden_hidden)
	z2 = tf.matmul(l1, w2)
	bn2 = batch_norm_wrapper(z2, is_training)
	l2 = tf.nn.sigmoid(bn2)

	#Softmax
	w3 = tf.Variable(w_hidden_output)
	b3 = tf.Variable(tf.zeros([10]))
	y = tf.nn.softmax(tf.matmul(l2, w3))


	cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
	train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
	correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(y_, 1))

	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	return (x, y_), train_step, accuracy, y, tf.train.Saver()




(x, y_), train_step, accuracy, _, saver = build_graph(is_training=True)

acc = []

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for i in tqdm.tqdm(range(10000)):
		batch = mnist.train.next_batch(60)
		train_step.run(feed_dict={x: batch[0], y_: batch[1]})
		if i % 50 is 0:
			res = sess.run([accuracy], feed_dict = {x: mnist.test.images, y_:mnist.test.labels})
			acc.append(res[0])

	saved_model = saver.save(sess, './temp-bn-save')

print ("Final accuracy: (%)", acc[-1]*100)



#Lets see the predictions of the model

reset_graph()
sess.close()

(x, y_), _, accuracy, y, saver = build_graph(is_training=False)

predictions = []
acc = 0

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	saver.restore(sess, './temp-bn-save')

	#Find the predictions and accuracy for the 100 examples
	for i in range(100):
		_x = [mnist.test.images[i]]
		_y = [mnist.test.labels[i]]
		
		pred, corr = sess.run([tf.arg_max(y, 1), accuracy],
			feed_dict={x: _x, y_: _y})
		acc += corr
		predictions.append(pred[0])

print("Predictions : ", predictions)
print ("Accuracy : (%)", acc)











