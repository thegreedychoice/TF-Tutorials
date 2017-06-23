# -*- coding: utf-8 -*-
"""
In this post, we’ll be building a no frills RNN that accepts a binary sequence X and uses it to 
predict a binary sequence Y. The sequences are constructed as follows-
Input sequence (X): At time step t, Xt has a 50% chance of being 1 (and a 50% chance of being 0). 
E.g., X might be [1, 0, 0, 1, 1, 1 … ].
Output sequence (Y): At time step t, Yt has a base 50% chance of being 1 (and a 50% base chance to be 0). 
The chance of Yt being 1 is increased by 50% (i.e., to 100%) if Xt−3 is 1, and decreased by 25% (i.e., to 25%)
if Xt−8 is 1. If both Xt−3 and Xt−8 are 1, the chance of Yt being 1 is 50% + 50% - 25% = 75%.
Thus, there are two dependencies in the data: one at t-3 (3 steps back) and one at t-8 (8 steps back).

This data is simple enough that we can calculate the expected cross-entropy loss for a trained RNN 
depending on whether or not it learns the dependencies:

If the network learns no dependencies, it will correctly assign a probability of 62.5% to 1,
for an expected cross-entropy loss of about 0.66.
If the network learns only the first dependency (3 steps back) but not the second dependency,
it will correctly assign a probability of 87.5%, 50% of the time, and correctly assign a probability 
of 62.5% the other 50% of the time, for an expected cross entropy loss of about 0.52.
If the network learns both dependencies, it will be 100% accurate 25% of the time, correctly assign 
a probability of 50%, 25% of the time, and correctly assign a probability of 75%, 50% of the time, 
for an expected cross extropy loss of about 0.45.

"""

import numpy as np
import tensorflow as tf 
#%matplotlib inline
import matplotlib.pyplot as plt 

# Global config variables
num_steps = 5 # number of truncated backprop steps 
batch_size = 200
num_classes = 2
state_size = 4 #hidden layer num of neuron units
eta = 0.1


#Generate the datset to be used by the model
def gen_data(size=1000000):
	X = np.array(np.random.choice(2, size=(size,)))
	Y = []
	for i in range(size):
		threshold = 0.5
		if X[i-3] == 1:
			threshold += 0.5
		if X[i-8] == 1:
			threshold -= 0.25
		if threshold > np.random.rand():
			Y.append(1)
		else:
			Y.append(0)

	return X, np.array(Y)


def gen_batch(raw_data, batch_size, num_steps):
    raw_x, raw_y = raw_data
    data_length = len(raw_x)

    # partition raw data into batches and stack them vertically in a data matrix
    batch_partition_length = data_length // batch_size
    data_x = np.zeros([batch_size, batch_partition_length], dtype=np.int32)
    data_y = np.zeros([batch_size, batch_partition_length], dtype=np.int32)
    for i in range(batch_size):
        data_x[i] = raw_x[batch_partition_length * i:batch_partition_length * (i + 1)]
        data_y[i] = raw_y[batch_partition_length * i:batch_partition_length * (i + 1)]
    # further divide batch partitions into num_steps for truncated backprop
    epoch_size = batch_partition_length // num_steps

    for i in range(epoch_size):
        x = data_x[:, i * num_steps:(i + 1) * num_steps]
        y = data_y[:, i * num_steps:(i + 1) * num_steps]
        yield (x, y)

def gen_epochs(n, num_steps):
    for i in range(n):
        yield gen_batch(gen_data(), batch_size, num_steps)	


#The Model
#define the placeholders
x = tf.placeholder(tf.int32, [batch_size, num_steps], name='input_placeholder')
y = tf.placeholder(tf.int32, [batch_size, num_steps], name='target_placeholder')
init_state = tf.zeros([batch_size, state_size])
#shape of x = (200, 5)
#shape of x_one_hot = (200, 5, 2)
#rnn_inputs is a list of num_steps(5) tensors 
#each tensor is of shape (200, 2) <--> (batch_size, num_classes)
#shape of rnn_inputs is (5, 200, 2)
x_one_hot = tf.one_hot(x, num_classes)
rnn_inputs = tf.unstack(x_one_hot, axis=1)

"""
Definition of rnn_cell
This is very similar to the way the __call__method of the tensorflow's
BasicRNNCell
"""

with tf.variable_scope('rnn_cell'):
    W = tf.get_variable('W', [num_classes + state_size, state_size])
    b = tf.get_variable('b', [state_size], initializer=tf.constant_initializer(0.0))

def rnn_cell(rnn_input, state):
    with tf.variable_scope('rnn_cell', reuse=True):
        W = tf.get_variable('W', [num_classes + state_size, state_size])
        b = tf.get_variable('b', [state_size], initializer=tf.constant_initializer(0.0))
    return tf.tanh(tf.matmul(tf.concat([rnn_input, state], 1), W) + b)


#Adding rnn_cells to the graph
#This is a simplified version of the "static_rnn" tensorflow function
state = init_state
rnn_outputs = []
for rnn_input in rnn_inputs:
	#shape of rnn_input : (200, 2) <--> (batch_size, num_classes)
	state = rnn_cell(rnn_input, state)
	#state is the output/hidden state of the RNN of shape (200,5)
	rnn_outputs.append(state)

#final_state of the unrolled RNN Layer
final_state = rnn_outputs[-1] 



#Predictions, Loss, training step
#Losses is similar to the "sequence_loss" TF API except that instead of 3-D Tensors, we are using 2-D Tensors

#logits and predictions
with tf.variable_scope('softmax'):
	W = tf.get_variable('W', [state_size, num_classes])
	b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))
logits = [tf.matmul(rnn_output, W) + b for rnn_output in rnn_outputs]
#shape of logits (5, 200, 2)

predictions = [tf.nn.softmax(logit) for logit in logits]


#Convert the Y placeholder into a list of labels
y_as_list = tf.unstack(y, num=num_steps, axis=1)

#losses and train steps
losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=logit)
				for logit, label in zip(logits, y_as_list)]

total_loss = tf.reduce_mean(losses)
train_step = tf.train.AdagradOptimizer(eta).minimize(total_loss)


#Now train the network

def train_network(num_epochs, num_steps, state_size=state_size, verbose=True):
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		training_losses = []
		for idx, epoch in enumerate(gen_epochs(num_epochs, num_steps)):
			training_loss = 0
			training_state = np.zeros((batch_size, state_size))

			if verbose:
				print("\nEpoch", idx)
			for step, (X,Y) in enumerate(epoch):
				tr_losses, training_loss_, training_state, _ = \
				sess.run([losses, total_loss, final_state, train_step],
				 feed_dict={x:X, y:Y, init_state:training_state})
				training_loss += training_loss_
				if step % 100 == 0 and step > 0:
					if verbose:
						print("Average Loss at step", step, 
							"for last 250 steps:", training_loss/100)
						training_losses.append(training_loss/100)
						training_loss = 0

	return training_losses



training_losses = train_network(10, num_steps)
"""
('Average Loss at step', 100, 'for last 250 steps:', 0.65592942237854002)
('Average Loss at step', 200, 'for last 250 steps:', 0.59760178387165075)
('Average Loss at step', 300, 'for last 250 steps:', 0.52569946497678754)
('Average Loss at step', 400, 'for last 250 steps:', 0.52180744349956509)
('Average Loss at step', 500, 'for last 250 steps:', 0.52026346832513815)
('Average Loss at step', 600, 'for last 250 steps:', 0.52193202465772626)
('Average Loss at step', 700, 'for last 250 steps:', 0.51931032478809358)
('Average Loss at step', 800, 'for last 250 steps:', 0.51976298689842226)
('Average Loss at step', 900, 'for last 250 steps:', 0.52000363707542419)
"""
plt.plot(training_losses)
plt.show()


"""
the network very quickly learns to capture the first dependency 
(but not the second), and converges to the expected cross-entropy loss of 0.52.
If we change hyperparameter where the num_steps=10; state_size=16: epochs = 10,
the network will learn the second dependency
as well and the cross entropy loss will come down to 0.45
"""













