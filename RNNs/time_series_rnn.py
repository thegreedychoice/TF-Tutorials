import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt 

#Hyperparameters
num_epochs = 100
total_series_length = 50000
truncated_backprop_length = 15
state_size = 4
num_classes = 2
echo_step = 3
batch_size = 5
num_batches = total_series_length//batch_size//truncated_backprop_length


#Collect Data
def generate_data():
	#0,1 50k samples
	x = np.array(np.random.choice(2, total_series_length, [0.5,0.5]))
	y = np.roll(x, echo_step) #The sequences of array moves 
	y[0:echo_step] = 0

	x = x.reshape((batch_size,-1))
	y = y.reshape((batch_size, -1))

	return (x,y)


data = generate_data()


#Generate Placeholders
batchX_placeholder = tf.placeholder(tf.float32, [batch_size, truncated_backprop_length])
batchY_placeholder = tf.placeholder(tf.int32, [batch_size, truncated_backprop_length])

init_state = tf.placeholder(tf.float32, [batch_size, state_size])

#Weights and biases1
W =  tf.Variable(np.random.rand(state_size+1, state_size), dtype=tf.float32)
b =  tf.Variable(np.zeros(1, state_size), dtype=tf.float32)

#Weights and biases2
W2 =  tf.Variable(np.random.rand(state_size, num_classes), dtype=tf.float32)
b2 =  tf.Variable(np.zeros(1, num_classes), dtype=tf.float32)

#Unstack this matrix into 1-D array
input_series = tf.unstack(batchX_placeholder, axis=1)
label_series = tf.unstack(batchY_placeholder, axis=1)


#Forward Pass
#state placeholder





