import numpy as np 
import tensorflow as tf 
import time
import matplotlib.pyplot as plt 
import os
import urllib2
import reader



#open the file and load the dataset
file_name = 'tinyshakespeare.txt'

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



def build_basic_rnn_graph_with_list(state_size=100, num_classes=vocab_size, batch_size=32, num_steps=200, eta=1e-4):
	"""
	This method builds the basic graph for a vanilla RNN networks of 200 timesteps 
	"""
	reset_graph()

	#Define the placeholders to hold the input data and labels
	#both x and y holds sequences of integars representing a sequence of characters
	x = tf.placeholder(tf.int32, [batch_size, num_steps], name='input_placeholder')
	y = tf.placeholder(tf.int32, [batch_size, num_steps], name='labels_placeholder')

	#Convert the data in x from Integar values to one-hot encoded values
	x_one_hot = tf.one_hot(x, num_classes) #shape is [32, 200, 65] where vocab_size = 65

	#Split x_one_hot into a list of 200 splits each representing the data of shape [32, 65] for individual timesteps
	rnn_inputs = [tf.squeeze(i, squeeze_dims=[1]) for i in tf.split(x_one_hot,num_steps, axis=1)] # each i is of shape [32, 1, 65]


	#Create the rnn Network
	cell = tf.contrib.rnn.BasicRNNCell(state_size)
	#shape of init_state = (32, 100)
	init_state = cell.zero_state(batch_size, tf.float32)
	#unroll the RNN Network
	rnn_outputs, final_state = tf.contrib.rnn.static_rnn(cell, rnn_inputs, initial_state=init_state)
	#rnn_outputs is a list of 200 outputs, each of shape [32, 100]


	#Declare the softmax layer parameters
	with tf.variable_scope('softmax'):
		W = tf.get_variable('W', [state_size, num_classes])
		b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))
	#Generate the logits which are basically the unnormalized probabilities of character in vocab for the prediction
	logits = [tf.matmul(rnn_output, W) + b for rnn_output in rnn_outputs] #list of 200 outputs, each of shape (32, 65)


	#Use the tensor flow legacy seq2seq model to calculate the cross-entrop loss
	#need to define a list of 200 weights each of [batch_size] 
	y_as_list = [tf.squeeze(i, squeeze_dims=[1]) for i in tf.split(y, num_steps, axis=1)]
	loss_weights = [tf.ones([batch_size]) for i in range(num_steps)]
	#losses holds loss for each example(sequence) in the batch, therefore a total of 32 losses
	losses = tf.contrib.legacy_seq2seq.sequence_loss_by_example(logits, y_as_list, loss_weights)
	total_loss = tf.reduce_mean(losses)
	train_step = tf.train.AdamOptimizer(eta).minimize(total_loss)

	return dict(
		x = x,
		y = y,
		init_state = init_state,
		final_state = final_state, 
		total_loss = total_loss,
		train_step = train_step
		)


"""
s = time.time()
build_basic_rnn_graph_with_list()
t = time.time()
print("It took ", t-s, "seconds to build the graph!")

"""

def build_multilayer_lstm_graph_with_list(state_size=100, num_classes=vocab_size, batch_size=32, num_steps=200, num_layers=3, eta=1e-4):
	"""
	This methods builds a graph with a network of num_layers of LSTMs cells
	"""

	reset_graph()

	x = tf.placeholder(tf.int32, [batch_size, num_steps], name='input_placeholder')
	y = tf.placeholder(tf.int32, [batch_size, num_steps], name='labels_placeholder')

	#Create a embeddings matrix to hold all the character emebeddings, each of size hidden_state, instead of one-hot ecoded vectors
	embeddings = tf.get_variable('embeddings_matrix', [num_classes, state_size])

	#Create inputs as a list of 200, for rnn each having 32 embedded vectors [32, 100]
	rnn_inputs = [tf.squeeze(i, squeeze_dims=[1]) for i in tf.split(tf.nn.embedding_lookup(embeddings, x), num_steps, axis=1)] 
	
	#Create a multi-stacked LSTM network 
	def lstm_cell(hidden_size):
		return tf.contrib.rnn.BasicLSTMCell(hidden_size, state_is_tuple=True)

	stacked_lstm = tf.contrib.rnn.MultiRNNCell(
		[lstm_cell(state_size) for _ in range(num_layers)],
		state_is_tuple=True)

	init_state = stacked_lstm.zero_state(batch_size, tf.float32)
	rnn_outputs, final_state = tf.contrib.rnn.static_rnn(stacked_lstm, rnn_inputs, initial_state=init_state)


	with tf.variable_scope('softmax'):
		W = tf.get_variable('W', [state_size, num_classes])
		b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))
	logits = [tf.matmul(rnn_output, W) + b for rnn_output in rnn_outputs]


	y_as_list = [tf.squeeze(i, squeeze_dims=[1]) for i in tf.split(y, num_steps, axis=1)]

	loss_weights = [tf.ones([batch_size]) for _ in range(num_steps)]
	losses = tf.contrib.legacy_seq2seq.sequence_loss_by_example(logits, y_as_list, loss_weights)
	total_loss = tf.reduce_mean(losses)
	train_step = tf.train.AdamOptimizer(eta).minimize(total_loss)

	return dict(
		x = x,
		y = y, 
		init_state=init_state,
		final_state=final_state,
		total_loss=total_loss,
		train_step = train_step
		)


"""
s = time.time()
build_multilayer_lstm_graph_with_list()
t = time.time()
print("It took ", t-s, "seconds to build the graph!")
"""


#Building graphs this way would take a lot of compile time and thus would be a better option to
#build the graph at the execution time using tensorflow's dynamic_run()



def build_multilayer_lstm_graph_with_dynamic_rnn(state_size=100, num_classes=vocab_size, batch_size=32, num_steps=200, num_layers=3, eta=1e-4):
	"""
	This methods builds a graph with a network of num_layers of LSTMs cells
	"""

	reset_graph()

	x = tf.placeholder(tf.int32, [batch_size, num_steps], name='input_placeholder')
	y = tf.placeholder(tf.int32, [batch_size, num_steps], name='labels_placeholder')

	#Create a embeddings matrix to hold all the character emebeddings, each of size hidden_state, instead of one-hot ecoded vectors
	embeddings = tf.get_variable('embeddings_matrix', [num_classes, state_size])

	#Get embeddings for all the input
	#shape of rnn_inputs = [batch_size, num_steps, state_size]
	rnn_inputs = tf.nn.embedding_lookup(embeddings, x)
	
	#Create a multi-stacked LSTM network 
	def lstm_cell(hidden_size):
		return tf.contrib.rnn.BasicLSTMCell(hidden_size, state_is_tuple=True)

	stacked_lstm = tf.contrib.rnn.MultiRNNCell(
		[lstm_cell(state_size) for _ in range(num_layers)],
		state_is_tuple=True)

	init_state = stacked_lstm.zero_state(batch_size, tf.float32)
	rnn_outputs, final_state = tf.nn.dynamic_rnn(stacked_lstm, rnn_inputs, initial_state=init_state)
	#shape of rnn_outputs [32, 200, 100]

	#But we need to reshape the rnn_outputs to do the matrix multiply fo each timestep
	#new shape of rnn_outputs [32*200, 100]
	rnn_outputs = tf.reshape(rnn_outputs, [-1, state_size])
	#print ("Shape of rnn_outputs :", rnn_outputs.shape)
	y_reshaped = tf.reshape(y, [-1])
	#print ("Shape  of labels : ", y_reshaped.shape)
	#shape of labels/y_reshaped : [6400] i.e., [batch_size * num_steps]
	
	with tf.variable_scope('softmax'):
		W = tf.get_variable('W', [state_size, num_classes])
		b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))
	logits = tf.matmul(rnn_outputs, W) + b
	#print ("Shape of logits : ", logits.shape)  
	#shape of logits : [6400, 65] i.e., [batch_size * num_steps, num_classes]

	#This loss function requires the labels to be of shape : [?] and each entry is 1-D int32 or int64 type
	#and logits to be of shape : [?, num_classes] where ? = batch_size * num_steps in this scenario
	losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=y_reshaped)
	#print ("Shapes of losses : ", loss.shape)
	total_loss = tf.reduce_mean(losses)
	train_step = tf.train.AdamOptimizer(eta).minimize(total_loss)

	return dict(
		x = x,
		y = y, 
		init_state=init_state,
		final_state=final_state,
		total_loss=total_loss,
		train_step = train_step
		)



def build_multilayer_lstm_graph_with_dynamic_rnn2(state_size=100, num_classes=vocab_size, batch_size=32, num_steps=200, num_layers=3, eta=1e-4):
	"""
	This methods builds a graph with a network of num_layers of LSTMs cells
	"""

	reset_graph()

	x = tf.placeholder(tf.int32, [batch_size, num_steps], name='input_placeholder')
	y = tf.placeholder(tf.int32, [batch_size, num_steps], name='labels_placeholder')

	#Create a embeddings matrix to hold all the character emebeddings, each of size hidden_state, instead of one-hot ecoded vectors
	embeddings = tf.get_variable('embeddings_matrix', [num_classes, state_size])

	#Get embeddings for all the input
	#shape of rnn_inputs = [batch_size, num_steps, state_size]
	rnn_inputs = tf.nn.embedding_lookup(embeddings, x)
	
	#Create a multi-stacked LSTM network 
	def lstm_cell(hidden_size):
		return tf.contrib.rnn.BasicLSTMCell(hidden_size, state_is_tuple=True)

	stacked_lstm = tf.contrib.rnn.MultiRNNCell(
		[lstm_cell(state_size) for _ in range(num_layers)],
		state_is_tuple=True)

	init_state = stacked_lstm.zero_state(batch_size, tf.float32)
	rnn_outputs, final_state = tf.nn.dynamic_rnn(stacked_lstm, rnn_inputs, initial_state=init_state)
	#shape of rnn_outputs [32, 200, 100]

	#But we need to reshape the rnn_outputs to do the matrix multiply fo each timestep
	#new shape of rnn_outputs [32*200, 100]
	rnn_outputs = tf.reshape(rnn_outputs, [-1, state_size])
	#print ("Shape of rnn_outputs :", rnn_outputs.shape)
	y_reshaped = tf.reshape(tf.one_hot(y, num_classes), [-1, num_classes])
	print ("Shape  of labels : ", y_reshaped.shape)
	#shape of labels/y_reshaped : [6400] i.e., [batch_size * num_steps]
	
	with tf.variable_scope('softmax'):
		W = tf.get_variable('W', [state_size, num_classes])
		b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))
	logits = tf.matmul(rnn_outputs, W) + b
	print ("Shape of logits : ", logits.shape)  
	#shape of logits : [6400, 65] i.e., [batch_size * num_steps, num_classes]

	#This loss function requires the labels and logits both to be of shape [?, num_classes]
	#where ? = batch_size * num_steps in this scenario
	losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=y_reshaped)
	#print ("Shapes of losses : ", loss.shape)
	total_loss = tf.reduce_mean(losses)
	train_step = tf.train.AdamOptimizer(eta).minimize(total_loss)

	return dict(
		x = x,
		y = y, 
		init_state=init_state,
		final_state=final_state,
		total_loss=total_loss,
		train_step = train_step
		)





g = build_multilayer_lstm_graph_with_dynamic_rnn2()
s = time.time()
train_network(g, 3)
print("It took ", time.time()-s, "seconds to train for 3 epochs.")










