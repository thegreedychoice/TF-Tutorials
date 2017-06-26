import tensorflow as tf 
import time
import matplotlib.pyplot as plt 
import os
import urllib2
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from RNNs import reader


class GRUCell(tf.contrib.rnn.RNNCell):
	""" This is similar to the tensorflow implementation of the basic GRU Cell"""
	def __init__(self, num_units):
		self._num_units = num_units

	@property
	def state_size(self):
		return self._num_units

	@property
	def output_size(self):
		return self._num_units

	def __call__(self, inputs, state, scope=None):
		"""
		This method performs the GRU Cell Gate operations and compute the new final state
		The following opertations are performed-
		r_t = sigmoid(U_r*H_t-1 + W_r*X_t + bias_r)
		u_t = sigmoid(U_u*H_t-1 + W_u*X_t + bias_u)
		candidate_state = tanh(U * (r*H_t-1) + W * X_t + bias)
		new_h = u_t * H_t-1 + (1-u_t) * candidate_state

		"""
		with tf.variable_scope(scope or type(self).__name__): #GRUCell
		#Now lets define the scope for tne reset and  update gates
			with tf.variable_scope("Gates"):
				#Start with the bias of 1.0 for both the gates such as not to reset or update
				r_u = tf.contrib.rnn.python.ops.core_rnn_cell_impl._linear([inputs, state], 2*self._num_units, True, 1.0)
				r_u = tf.nn.sigmoid(r_u)

				#r_u contains the scalar values of shape [state_size, 2] for both reset and update gates
				r, u = tf.split(r_u, 2, axis=1)

			#Now define another scope for the computed/candidate hidden state
			with tf.variable_scope("Candidate_State"):
				h_c = tf.nn.tanh(tf.contrib.rnn.python.ops.core_rnn_cell_impl._linear([inputs, r*state], self._num_units, True, 1.0))

			#calculate the output of the cell
			new_h = u * state + (1-u) * h_c
		return new_h, new_h




class CustomCell(tf.contrib.rnn.RNNCell):
	"""
	This custom cell will perform the operations differently than the GRUCell defined above.
	Instead of using W * X_t in the computation of the candidate state, we would use a weighted average of 
	W1 * X_t, W_2 *X_t,..., W_n * X_t for some n. In other words we will replace W_n * X_t with sum(lambda_i * W_i * X)
	where lamba = softmax(W_avg * X_t + U_avg * H_t-1 + b)
	The idea is that we might benefit from treating the input differently in different scenarios (we may want to treat verbs differently than nouns).
	"""
	def __init__(self, num_units, num_weights):
		"""
		The num_weights defines the number of weight matrices that needs to be initialized
		"""
		self._num_weights = num_weights
		self._num_units = num_units

	@property
	def state_size(self):
		return self._num_units

	@property
	def output_size(self):
		return self._num_units

	def __call__(self, inputs, state, scope=None):
		with tf.variable_scope(scope or type(self).__name__): 
			with tf.variable_scope("Gates"):
								#Start with the bias of 1.0 for both the gates such as not to reset or update
				r_u = tf.contrib.rnn.linear([inputs, state], 2*self._num_units, True, 1.0)
				r_u = tf.nn.sigmoid(r_u)

				#r_u contains the scalar values of shape [state_size, 2] for both reset and update gates
				r, u = tf.split(r_u, 2, axis=1)

				with tf.variable_scope("Candidate"):
					lambdas = tf.contrib.rnn._linear([inputs, state], self._num_weights, True)
					lambdas = tf.split(tf.nn.softmax(lambdas), self._num_weights, axis=1)

					W_s = tf.get_variable("Ws", shape = [self._num_weights, inputs.get_shape()[1], self._num_units])
					W_s = [tf.squeeze(i) for i in tf.split(W_s, self._num_weights, axis=0)]

					candidate_inputs = []

					for idx, W in enumerate(W_s):
						candidate_inputs.append(tf.matmul(inputs, W) * lambdas[idx])

					Wx = tf.add_n(candidate_inputs)

					c = tf.nn.tanh(Wx + tf.contrib.rnn._linear([r*state], self._num_units, True, scope="second"))

			new_h = u * state + (1-u) * c

			return new_h, new_h

def reset_graph():
	"""
	This method resets the Tensorflow Session's Default Graph
	"""
	if 'sess' in globals() and sess:
		sess.close()
	tf.reset_default_graph()

def train_network(g, num_epochs, num_steps = 200, batch_size = 32, verbose = True, save = False):
	tf.set_random_seed(2345)
	training_losses = []
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for idx, epoch in enumerate(gen_epochs(num_epochs, num_steps, batch_size)):
			#This outer loop runs for num_epochs time and 'epoch' is an interator for the dataset
			training_loss = 0
			training_state = None
			steps = 0

			for X, Y in epoch:
				#This method will loop through all the batches in the dataset for each epoch
				steps += 1

				feed_dict = {g['x'] : X, g['y']:Y}
				if training_state is not None:
					feed_dict[g['init_state']] = training_state

				training_loss_, training_state, _ = sess.run([g['total_loss'], 
														g['final_state'], 
														g['train_step']],
														 feed_dict)

				training_loss += training_loss_

			if verbose:
				print "Average Training Loss for Epoch {0} : {1}".format(idx, training_loss/steps)

			training_losses.append(training_loss/steps)

		if isinstance(save, str):
			g['saver'].save(sess, save)

	return training_losses

#open the file and load the dataset
file_name = 'data/tinyshakespeare.txt'

with open(file_name, 'r') as f:
	raw_data = f.read()
	print("Data Length", len(raw_data))


vocab = set(raw_data)
vocab_size = len(vocab)
#Generate the Index to Char & Char to Index Dictionary for encoding/decoding
idx_to_char = dict(enumerate(vocab))
char_to_idx = dict(zip(idx_to_char.values(), idx_to_char.keys()))

#The characters are now represented as integars
data = [char_to_idx[c] for c in raw_data]
del raw_data

def gen_epochs(num_epochs, num_steps, batch_size):
	"""
	This method returns the 'num_epochs' number of interators each of shape (batch_size, num_steps)
	one after the other.
	"""
	for i in range(num_epochs):
		yield reader.ptb_iterator(data, batch_size, num_steps)


def build_multilayer_graph_with_custom_cell(
	cell_type = None,
	num_weights_for_custom_cell = 5, 
	state_size = 100,
	num_classes = vocab_size, 
	batch_size = 32,
	num_steps = 200,
	num_layers = 3,
	eta = 1e-4):
	
	reset_graph()

	x = tf.placeholder(tf.int32, [batch_size, num_steps], name="input_placeholder")
	y = tf.placeholder(tf.int32, [batch_size, num_steps], name="labels_placeholder")

	embeddings = tf.get_variable("embedding_matrix", [num_classes, state_size])

	rnn_inputs = tf.nn.embedding_lookup(embeddings, x)

	def lstm_cell(hidden_size):
		return tf.contrib.rnn.BasicLSTMCell(hidden_size, state_is_tuple=True)

	def gru_cell(hidden_size):
		return tf.contrib.rnn.GRUCell(hidden_size)

	if cell_type == 'Custom':
		cell = tf.contrib.rnn.MultiRNNCell([CustomCell(state_size, num_weights_for_custom_cell) for _ in range(num_layers)]) 
	elif cell_type == 'LSTM':
		cell = tf.contrib.rnn.MultiRNNCell([lstm_cell(state_size) for _ in range(num_layers)])
	elif cell_type == 'GRU':
		cell = tf.contrib.rnn.MultiRNNCell([gru_cell(state_size) for _ in range(num_layers)])
	else:
		cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicRNNCell(state_size) for _ in range(num_layers)])


	init_state = cell.zero_state(batch_size, tf.float32)
	rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, rnn_inputs, initial_state=init_state)

	with tf.variable_scope("softmax"):
		W = tf.get_variable('W', [state_size, num_classes])
		b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))

	rnn_outputs = tf.reshape(rnn_outputs, [-1, state_size])
	y_reshaped = tf.reshape(y, [-1])

	logits = tf.matmul(rnn_outputs, W) + b
	total_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_reshaped))
	train_step = tf.train.AdamOptimizer(eta).minimize(total_loss)

	return dict(
		x = x,
		y = y,
		init_state = init_state, 
		final_state = final_state,
		total_loss = total_loss, 
		train_step = train_step)




g = build_multilayer_graph_with_custom_cell(cell_type='GRU', num_steps=30)
s = time.time()
train_network(g, 5, num_steps=30)
e = time.time()
print("It took ", e-s, "seconds to train for ", 5, " epochs")











	