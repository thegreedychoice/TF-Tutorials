import numpy as np
import tensorflow as tf


sample_input = tf.constant([[1,2,3,4,3,2],[3,2,2,2,2,2]], dtype=tf.float32)
LSTM_CELL_SIZE = 3 #Number of timesteps = 3

lstm_cell = tf.contrib.rnn.BasicLSTMCell(LSTM_CELL_SIZE, state_is_tuple=True)
state = (tf.zeros([2, LSTM_CELL_SIZE]),)*2 

with tf.variable_scope('lstm_basics') as scope:
	output, state_new = lstm_cell(sample_input, state)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

print "Sample Input :"
print sess.run(sample_input)
print ""

print "States :"
print sess.run(state)
print ""

print "Cell State (c):"
print sess.run(state_new[0])
print ""

print "Output (h):"
print sess.run(state_new[1])
print ""

