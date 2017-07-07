import tensorflow as tf 
import time
import numpy as np
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
			print "The graph was saved!"

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



"""
g = build_multilayer_graph_with_custom_cell(cell_type='GRU', num_steps=30)
s = time.time()
train_network(g, 5, num_steps=30)
e = time.time()
print("It took ", e-s, "seconds to train for ", 5, " epochs")
"""


def ln(tensor, scope=None, epsilon = 1e-5):
	""" Layer normalizes 2-D tensor along its 2nd dimension"""
	assert(len(tensor.get_shape()) == 2)
	mean, var = tf.nn.moments(tensor, axes=[1], keep_dims=True)
	if not isinstance(scope, str):
		scope = ''
	with tf.variable_scope(scope+'layer_norm'):
		scale = tf.get_variable('scale',
		 shape=[tensor.get.shape()[1]], 
		 initializer=tf.constant_initializer(1))
		shift = tf.get_variable('shift', 
		shape=[tensor.get_shape()[1]], 
		initializer=tf.constant_initializer(0))
	LN_initial = (tensor - mean) /tf.sqrt(var + epsilon)
	LN_final = scale * LN_initial + shift

	return LN_final


class LayerNormalizedLSTMCell(tf.contrib.rnn.RNNCell):
	"""
	Adapted from tensorflow's version of BasicLSTMCell with Layer Normalization.
	We add layer normalization to the output of each gate of the LSTM Cell
	"""

	def __init__(self, num_units, forget_bias=1.0, activation=tf.nn.tanh):
		#num of hidden neurons
		self._num_units = num_units
		self._forget_bias = forget_bias
		self._activation = activation

	@property
	def state_size(self):
		return tf.contrib.rnn.LSTMStateTuple(self._num_units, self._num_units)

	@property
	def output_size(self):
		return self._num_units

	def __call__(self, inputs, state, scope=None):
		"""
		The implementation of the basic LSTM Cell with layer normalization
		"""
		with tf.variable_scope(scope or type(self).__name__):
			c, h = state

			#change bias argument to False since LN will add bias via shift
			concat = tf.contrib.rnn_cell._linear([inputs, h], 4 * self._num_units, False)

			i, j, f, o = tf.split(concat, 4, axis=1)

			#add normalization to each gate
			i = ln(i, scope='i/')
			j = ln(j, scope='j/')
			f = ln(f, scope='f/')
			o = ln(o, scope='o/')

			new_c = (c * tf.nn.sigmoid(f + self._forget_bias) + tf.nn.sigmoid(i) * self._activation(j))

			#add layer normalization to the new hidden state
			new_h = self._activation(ln(new_c, scope='new_h/')) * tf.nn.sigmoid(o)
			new_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)

			return new_h, new_state




def build_graph_with_custom_cell(
	cell_type = None,
	num_weights_for_custom_cell = 5, 
	state_size = 100,
	num_classes = vocab_size, 
	batch_size = 32,
	num_steps = 200,
	num_layers = 3,
	build_with_dropout=True,
	eta = 1e-4):
	
	reset_graph()

	x = tf.placeholder(tf.int32, [batch_size, num_steps], name="input_placeholder")
	y = tf.placeholder(tf.int32, [batch_size, num_steps], name="labels_placeholder")

	dropout = tf.constant(1.0)

	embeddings = tf.get_variable("embedding_matrix", [num_classes, state_size])

	rnn_inputs = tf.nn.embedding_lookup(embeddings, x)

	def lstm_cell(hidden_size):
		return tf.contrib.rnn.BasicLSTMCell(hidden_size, state_is_tuple=True)

	def gru_cell(hidden_size):
		return tf.contrib.rnn.GRUCell(hidden_size)

	if build_with_dropout == False:
		if cell_type == 'Custom':
			cell = tf.contrib.rnn.MultiRNNCell([CustomCell(state_size, num_weights_for_custom_cell) for _ in range(num_layers)]) 
		elif cell_type == 'LSTM':
			cell = tf.contrib.rnn.MultiRNNCell([lstm_cell(state_size) for _ in range(num_layers)])
		elif cell_type == 'GRU':
			cell = tf.contrib.rnn.MultiRNNCell([gru_cell(state_size) for _ in range(num_layers)])
		elif cell_type == 'LN_LSTM':
			cell = tf.contrib.rnn.MultiRNNCell([LayerNormalizedLSTMCell(state_size) for _ in range(num_layers)])
		else:
			cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicRNNCell(state_size) for _ in range(num_layers)])

	else:

		if cell_type == 'Custom':
			cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(
				CustomCell(state_size, num_weights_for_custom_cell), input_keep_prob=dropout)
				 for _ in range(num_layers)]) 
		elif cell_type == 'LSTM':
			cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(
				lstm_cell(state_size), input_keep_prob=dropout)
				 for _ in range(num_layers)])
		elif cell_type == 'GRU':
			cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(
				gru_cell(state_size), input_keep_prob=dropout)
				 for _ in range(num_layers)])
		elif cell_type == 'LN_LSTM':
			cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(
				LayerNormalizedLSTMCell(state_size), input_keep_prob=dropout)
				 for _ in range(num_layers)])
		else:
			cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(
				tf.contrib.rnn.BasicRNNCell(state_size), input_keep_prob=dropout)
				 for _ in range(num_layers)])

		cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=dropout)



	init_state = cell.zero_state(batch_size, tf.float32)
	rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, rnn_inputs, initial_state=init_state)

	with tf.variable_scope("softmax"):
		W = tf.get_variable('W', [state_size, num_classes])
		b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))

	rnn_outputs = tf.reshape(rnn_outputs, [-1, state_size])
	y_reshaped = tf.reshape(y, [-1])

	logits = tf.matmul(rnn_outputs, W) + b
	predictions = tf.nn.softmax(logits)
	total_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_reshaped))
	train_step = tf.train.AdamOptimizer(eta).minimize(total_loss)

	return dict(
		x = x,
		y = y,
		init_state = init_state, 
		final_state = final_state,
		total_loss = total_loss, 
		train_step = train_step,
		preds = predictions,
		saver = tf.train.Saver()
		)



g = build_graph_with_custom_cell(cell_type='GRU', num_steps=80)
s = time.time()
losses = train_network(g, num_epochs=1, num_steps=80, save="saves/GRU_20_epochs")
e = time.time()
print("It took ", e-s, "seconds to train for ", 5, " epochs")
print("The average loss on the final epoch was : ", losses[-1])


"""
Now lets generate characters using these RNN models we have defined
"""

def gen_characters(g, checkpoint, num_chars, prompt='A', pick_top_chars=None):
	"""
	Accepts the current character and a checkpoint to restore from
	"""

	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())
		g['saver'].restore(sess, checkpoint)

	state = None
	current_char = char_to_idx[prompt]
	chars = [current_char]
	for i in range(num_chars):
		#prepare the feed_dictionary
		if state is not None:
			feed_dict = {g['x']:[[current_char]], g['init_state']:state}
		else:
			feed_dict = {g['x']:[[current_char]]}

		#find the predictions distribution for the next char
		preds, state = sess.run([g['preds'], g['final_state']], feed_dict)
		#Preds holds the probability distribution of the next character

		if pick_top_chars is not None:
			p = np.squeeze(preds)
			p[np.argsort(p)[:-pick_top_chars]] = 0
			#Normalize the probabilities for the top picks
			p = p/np.sum(p)
			#randomly choose one of the character from the picked ones
			current_char = np.random.choice(vocab_size, 1, p=p)[0]
		else:
			current_char = np.random.choice(vocab_size, 1, p=np.squeeze(preds))[0]

		chars.append(current_char)

	#convert these character indexes to characters

	chars = map(lambda x:idx_to_char[x], chars)

	print "".join(chars)

	return "".join(chars)




g = build_graph_with_custom_cell(cell_type='GRU', num_steps=1, batch_size=1)

gen_characters(g, "saves/GRU_20_epochs", 750, prompt='A', pick_top_chars=5)





	