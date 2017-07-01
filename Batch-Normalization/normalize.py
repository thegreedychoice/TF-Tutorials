"""
In this program we will add batch normalization to a basic fully-connected neural network that has two hidden layers of 100 neurons each
and show how it can have a positive impact on the training time and accuracy.
Also we will see how using our model built on batch mean and variance for normalization during training, 
is a bad idea to use for testing later on. Intuition gets better if we test using a single test example
"""


import numpy as np 
import tensorflow as tf, tqdm
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt 

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#Generate predetermined random weights so that the networks are similarly initialized
w_input_hidden = tf.constant(np.random.normal(size=(784,100)).astype(np.float32))
w_hidden_hidden = tf.constant(np.random.normal(size=(100,100)).astype(np.float32))
w_hidden_output = tf.constant(np.random.normal(size=(100, 10)).astype(np.float32))

#Intialize small epsilon value for Batch Normalization transform 
epsilon = 1e-3

eta = 0.01

#Lets define the placeholders for input and targets
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

#Create scope for Layer 1 without Batch Normalization
with tf.variable_scope('Layer1_wo_BN'):
	w1 = tf.get_variable('w', initializer=w_input_hidden)
	b1 = tf.get_variable('B', initializer=tf.zeros([100]))
	z1 = tf.matmul(x, w1) + b1
	act1 = tf.nn.sigmoid(z1)

#Create another scope for the same Layer 1 but with batch normalization
with tf.variable_scope('Layer1_w_BN'):
	w1_BN = tf.get_variable('w', initializer=w_input_hidden)
	#The pre-batch normalization bias is omitted since its effect would be cancelled by subtracting the batch mean later.
	#instead the role of this bias is performed by the new beta variable used to shft the normalizaed values.
	z1_BN = tf.matmul(x, w1_BN)

	#Find the batch mean and variance
	batch_mean1, batch_var1 = tf.nn.moments(z1_BN, [0])

	#Apply the initial batch normalizing transform to z1_BN
	#This normalizes the net values to have a normal distribution with mean 0 and variance 1
	z1_hat = (z1_BN -  batch_mean1) / tf.sqrt(batch_var1 + epsilon)

	#Create two new paramters to kinda un-normalize or scale and shift the normalized values from the initial transform
	scale1 = tf.get_variable('scale', initializer=tf.ones([100]))
	beta1 =  tf.get_variable('beta', initializer=tf.zeros([100]))

	#Scale and shift to obtain the final output of the batch normalization
	#This brings back the representational power of the hidden layers of the network
	z1_BN = scale1 * z1_hat + beta1
	#Do a non-linear activation on this final output value
	act1_BN = tf.nn.sigmoid(z1_BN)



#Create scope for Layer 2 without Batch Normalization
with tf.variable_scope('Layer2_wo_BN'):
	w2 = tf.get_variable('w', initializer=w_hidden_hidden)
	b2 = tf.get_variable('b', initializer=tf.zeros([100]))
	z2 = tf.matmul(act1, w2) + b2
	act2 = tf.nn.sigmoid(z2)

#Create another scope for the same Layer 2 but with batch normalization
#Instead of defining BN operations, we make use of tensorflow op's for the same
with tf.variable_scope('Layer2_w_BN'):
	w2_BN = tf.get_variable('w', initializer=w_hidden_hidden)
	#The pre-batch normalization bias is omitted since its effect would be cancelled by subtracting the batch mean later.
	#instead the role of this bias is performed by the new beta variable used to shft the normalizaed values.
	z2_BN_ = tf.matmul(act1_BN, w2_BN)

	#Find the batch mean and variance
	batch_mean2, batch_var2 = tf.nn.moments(z2_BN_, [0])

	#Create two new paramters to scale and shift the normalized values from the initial transform
	scale2 = tf.get_variable('scale', initializer=tf.ones([100]))
	beta2 =  tf.get_variable('beta', initializer=tf.zeros([100]))

	#Scale and shift to obtain the final output of the batch normalization
	#This brings back the representational power of the hidden layers of the network
	z2_BN = tf.nn.batch_normalization(z2_BN_, batch_mean2, batch_var2, beta2, scale2, epsilon)
	#Do a non-linear activation on this final output value
	act2_BN = tf.nn.sigmoid(z2_BN)


#Create the scope for softmax
with tf.variable_scope('softmax_wo_BN'):
	w3 = tf.get_variable('w', initializer=w_hidden_output)
	b3 = tf.get_variable('b', initializer=tf.zeros([10]))
	y = tf.nn.softmax(tf.matmul(act2, w3) + b3)

with tf.variable_scope('softmax_w_BN'):
	w3_BN = tf.get_variable('w', initializer=w_hidden_output)
	b3_BN = tf.get_variable('b', initializer=tf.zeros([10]))
	y_BN = tf.nn.softmax(tf.matmul(act2_BN, w3_BN) + b3_BN)

#Loss, optimizer and predictions
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
cross_entropy_BN = -tf.reduce_sum(y_*tf.log(y_BN))

train_step = tf.train.GradientDescentOptimizer(eta).minimize(cross_entropy)
train_step_BN = tf.train.GradientDescentOptimizer(eta).minimize(cross_entropy_BN)

correct_predictions = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32)) * 100

correct_predictions_BN = tf.equal(tf.argmax(y,1), tf.argmax(y_BN,1))
accuracy_BN = tf.reduce_mean(tf.cast(correct_predictions_BN, tf.float32)) * 100


z2s, z2_BNs, accs, acc_BNs = [], [], [], []

num_epochs = 40000
batch_size = 60
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
for i in tqdm.tqdm(range(num_epochs)):
	batch = mnist.train.next_batch(batch_size)
	train_step.run(feed_dict={x:batch[0], y_:batch[1]})
	train_step_BN.run(feed_dict={x:batch[0], y_:batch[1]})

	if i % 50 is 0:
		res = sess.run([accuracy, accuracy_BN, z2, z2_BN], feed_dict={x:mnist.test.images, y_:mnist.test.labels})
		accs.append(res[0])
		acc_BNs.append(res[1])
		#Record the mean values of z2 over the entire test set
		z2s.append(np.mean(res[2], axis=0))  
		#Record the mean values of normalized z2 over the entire test set 
		z2_BNs.append(np.mean(res[3], axis=0))

z2s, z2_BNs, accs1, acc_BNs1 = np.array(z2s), np.array(z2_BNs), np.array(accs), np.array(acc_BNs) 


#This plot will show how accuracy is effected by batch normalization
fig, ax = plt.subplots()
x = range(0, len(accs1)*50, 50)
ax.plot(x, accs1, label="Without BN")
ax.plot(x, acc_BNs1, label="With BN")
ax.set_xlabel('Training Steps')
ax.set_ylabel('Accuracy %')
ax.set_ylim([90, 100])
ax.set_title('Batch Normalization Accuracy')
ax.legend(loc=4)
#plt.show()


#This graph will show the distribution of inputs to the sigmoid activation over time
#It will show how much the input values vary representing the noise and how
#batch normalization is able to change that.
#We will show the distribution of the first n neurons of the second layer.
#Distribution at each neuron will be displayed in each row from 1-n

n = 5
fig, axes = plt.subplots(n, 2, figsize=(6,12))
fig.tight_layout()

for i, ax in enumerate(axes):
	ax[0].set_title("Without BN")
	ax[1].set_title("With BN")
	#i denotes ith neuron of the second layer
	ax[0].plot(z2s[:,i])
	ax[1].plot(z2_BNs[:,i])

plt.show()

#Making predictions with the model using the mean batch and avariance to normalize inputs to activations
#Using a single example to test at a time, will lead us to the same output everytime since the 
#batch mean would be the same and the normalized value would be 0.
predictions = []
correct = 0

for i in range(100):
	pred, corr = sess.run([tf.arg_max(y_BN, 1), accuracy_BN], 
		feed_dict={x: mnist.test.images[i], y_: mnist.test.labels[i]})
	correct += corr
	predictions.append(pred[0])

print("Predictions: ", predictions)
print("Accuracy(%):", correct)






