"""
This program will perform language modelling on the PEN Tree Dataset.
Language Modelling is the the task of assiging probabilties to a set of words (sentence)
in a given context. Based on these probability assigments, it can be used to predict word
given a certain input and a context.
Language modelling helps greatly in Text-generation, Speech Recognition, Image Captioning etc.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np 
import tensorflow as tf 

#Loading the PEN Tree Dataset
#!wget -q -O /resources/data/ptb.zip https://ibm.box.com/shared/static/z2yvmhbskc45xd2a9a4kkn6hg4g4kj5r.zip
#!unzip -o /resources/data/ptb.zip -d /resources/
#!cp /resources/ptb/reader.py .
import reader

#Download and extract the simple examples dataset
#!wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz 
#!tar xzf simple-examples.tgz -C /resources/data/


#Intialize hyperparamters for the model
#Intial weight scale
init_scale = 0.1
#Initial Learning rate
eta = 1.0
#Maximum possible norm for gradient (For gradient clipping in case of exploding gradients)
max_grad_norm = 5
#The number of layers in the model
num_layers = 2
#Num of times, i.e., number of recurrence layers when RNN unit is unfolded
num_steps = 20
#Num of hidden neurons in each RNN unit 
hidden_size = 200
#Maximum num of epochs trained with initial learning rate
max_epoch_intial = 4
#Total num of epochs in the training
max_epoch_total = 13
#Proabibility of keeping data in the dropout layer
#At probability 1, we ignore the Dropout layer wrapping
keep_prob = 1
#The decay for learning rate
eta_decay = 0.5
#Batch size of each batch of data
batch_size = 30
#The size of our vocabulary
vocab_size = 10000
#Trainig flag to seperate from testing
is_Training = 1
#Data directory for our dataset
data_dir = "data/resources/data/simple-examples/data/"


"""
Read the following for better understanding
Network Architecture:

-The number of LSTM cells are 2, i.e, One layer of LSTM is stacked upon another
and the outputs at each timestep of the first LSTM cell will serve as input to the second cell.
-Each LSTM cell when unfolded will have 20 timesteps, i.e., the recurrence step is 20
- The structure is following:

200 Input Units ---[200*200](Weight)---> 200 Hidden Units(First Layer) ---[200*200](Weight)
--->200 Hidden Units(Second Layer)---[200*200](Weight)--->200 Unit Output

Hidden Layer:
Each Hidden Layer consists of 200 hidden neurons which is also equal to the dimensionality
of the word embeddings (input) and the output vector

Input Layer:
The Network has 200 input units
Each input vector is word embedding with dimensions e, where in this case e=200
Input Shape is [batch_size, num_steps, n_input] = [30*20*200] after embedding 
and then 20*[30*200]
"""


"""
The training dataset is list of words represented by numbers N=929589 numbers
e.g., [9971, 9972, 9974, 9975, .....]
We read data as minibatch of size b = 30 (30 sentences) and assume each sentences has 
h = 20 words as number of recurrence steps = 20
So, number of iterations needed to perform will be (N/b*h) + 1 = 1549
Each batch data read from train dataset is of size 600 words [30*20]

"""

def lstm_cell(lstm_size):
  return tf.contrib.rnn.BasicLSTMCell(lstm_size)
#Lets start an interactive session
sess = tf.InteractiveSession()


#Read the dataset and seperates it into training, validation and testing data
raw_data = reader.ptb_raw_data(data_dir)
train_data, validation_data, test_data, _ = raw_data

#Read the first mini batch 
iterator = reader.ptb_iterator(train_data, batch_size, num_steps)
first_batch = iterator.next()
_x = first_batch[0]
_y = first_batch[1]

"""
Shape of x and y = (30, 20)
X[0] = [9970 9971 9972 9974 9975 9976 9980 9981 9982 9983 9984 9986 9987 9988 9989
 9991 9992 9993 9994 9995]
Y[0] = [9971 9972 9974 9975 9976 9980 9981 9982 9983 9984 9986 9987 9988 9989 9991
 9992 9993 9994 9995 9996]
Y is just the same sentence shifted one word forward
"""

size = hidden_size

#Define placeholders for the input and targets
x = tf.placeholder(tf.int32, [batch_size, num_steps]) #[30*20]
y = tf.placeholder(tf.int32, [batch_size, num_steps])

#define a feed dictionary to feed the placeholders with our first mini-batch
feed_dict = {x:_x, y:_y}

#e.g., we can use it to feed the x placeholder
sess.run(x, feed_dict)

#Create the 2-layer stacked LSTM network
lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=0.0)
#lstm_cells = []
#for _ in range(num_layers):
#	lstm_cells.append(tf.contrib.rnn.LSTMCell(hidden_size))


stacked_lstm = tf.contrib.rnn.MultiRNNCell(
    [lstm_cell(hidden_size) for _ in range(num_layers)])
#stacked_lstm = tf.contrib.rnn.MultiRNNCell(lstm_cells)


#Each LSTM cell has two states Cell State(c) and Memory State(h) and each is of
#shape (batch_size, hidden_size)
#Create the state matrices and intialize it to zero
_intial_state = stacked_lstm.zero_state(batch_size, tf.float32)

sess.run(_intial_state, feed_dict)

#Now we need to create the word embeddings using the tensorflow embedding function
#This will hold the embeddings for each unqiue word in the dataset(vocabulary of datset)
embedding = tf.get_variable("embedding", [vocab_size, hidden_size]) #10000* 200

#sess.run(tf.global_variables_initializer())
#Creates a word embedding based on the first batch from our dataset, initialized 
#using feed dictionary
#sess.run(embedding, feed_dict)
#print(sess.run(embedding, feed_dict).shape)
#(10000, 200)

#To get the embedding for our input data we need to use tensor-flow's embedding_lookup
#embedding_lookup goes to each row of _x and for each word in that row finds the word embedding
# in variable 'embedding'. Therefore returns (20, 200) for each row and given we have 30 rows(batch_size)
#the function returns a matrix of embeddings of shape (30,20,200)

inputs = tf.nn.embedding_lookup(embedding, _x) #returns embeddings for first batch 

#print(inputs.shape)
#(30, 20, 200)

#print(sess.run(inputs[0], feed_dict))


"""
Next step is to construct a RNN with 2 LSTM Layers(stacked_lstm instance) 
which can be then unfolded into 20 timesteps
We can use tensor flow's dynamicrnn() which creates an unfolded RNN
It accepts the input of shape (batch_size, num_steps, n_input) i.e., (30,20,200)
and returns the outputs at each timestep and a final state(at timestep 20)
The outputs and states are different from the LSTM's cell state
Shape of outputs : (30, 20, 200)
"""

outputs, new_state = tf.nn.dynamic_rnn(stacked_lstm, inputs, initial_state=_intial_state)

#print(outputs)
sess.run(tf.global_variables_initializer())
















