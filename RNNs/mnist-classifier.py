import warnings
warnings.filterwarnings('ignore')
import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(".", one_hot=True)

train_images = mnist.train.images 
train_labels = mnist.train.labels 
test_images = mnist.test.images 
test_labels = mnist.test.labels 


ntrain = train_images.shape[0]
ntest = test_images.shape[0]
dim = train_images.shape[1]
nclasses = train_labels.shape[1]
print "Train Images: ", train_images.shape
print "Train Labels  ", train_labels.shape
print
print "Test Images:  " , test_images.shape
print "Test Labels:  ", test_labels.shape


#Visualizing the mnist sample
samplesIdx = [100, 101, 102]  #<-- You can change these numbers here to see other samples

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()

ax1 = fig.add_subplot(121)
ax1.imshow(test_images[samplesIdx[0]].reshape([28,28]), cmap='gray')

xx, yy = np.meshgrid(np.linspace(0,28,28), np.linspace(0,28,28))
X = xx
Y = yy
Z = 100*np.ones(X.shape)

img = test_images[77].reshape([28,28])
ax = fig.add_subplot(122, projection='3d')
ax.set_zlim((0,200))

offset=200
for i in samplesIdx:
    img = test_images[i].reshape([28,28]).transpose()
    ax.contourf(X, Y, img, 200, zdir='z', offset=offset, cmap="gray")
    offset -= 100

    ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

#plt.show()


for i in samplesIdx:
    print "Sample: {0} - Class: {1} - Label Vector: {2} ".format(i, np.nonzero(test_labels[i])[0], test_labels[i])

n_input = 28 # MNIST data input (img shape: 28*28) , length of each input sequence
n_steps = 28 # timesteps
n_hidden = 128 # hidden layer num of features
n_classes = 10 # MNIST total classes (0-9 digits)


eta = 0.001
training_iters = 100000
batch_size = 100
display_step = 10


x = tf.placeholder(dtype="float", shape=[None, n_steps, n_input], name="x")
y = tf.placeholder(dtype="float", shape=[None, n_classes], name="y")


#Weights and biases for output layer 
weights = {
	'out' : tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
	'out' : tf.Variable(tf.random_normal([n_classes]))
}

#Define the Network
lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)

#initial state
#initial_state = (tf.zeros([1,n_hidden]),)*2
outputs, states = tf.nn.dynamic_rnn(lstm_cell, inputs=x, dtype=tf.float32)

def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # The Current Data Input Shape of x is (batch_size, n_steps, n_inputs)
    #Each sample image is of size (28,28) pixels
    #The RNN layer will accept each row as a input sequence and the number of rows 
    #basically defines the number of timesteps
    #For example, if each sample size is (20,28), then the length of input sequence is 28
    #the number of timesteps is 20

    # Define a lstm cell with tensorflow
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
   

    # Get lstm cell output
    # outputs will hold the outputs/hidden states at each time step from 0 to 27 for each sample
    # The shape of outputs is (batch_size, n_inputs, n_hidden)
    outputs, states = tf.nn.dynamic_rnn(lstm_cell, inputs=x, dtype=tf.float32)

    # Get lstm cell output
    #outputs, states = lstm_cell(x , initial_state)

    #Output of the lstm cell is the final hidden states or the hidden state at time t=27
    #First split the outputs array and and get all the hidden states and then access the last one
    #We need to reshape the output (batch_size, n_hidden) 
    #i.e., each sample is of shape (128, ?)
    output = tf.reshape(tf.split(outputs, 28, axis=1, num='None', name='split')[-1], [-1, 128])
    #The final output of this method RNN is linear activation of the network
    # output [100x128] x  weight [128, 10] + [], where 100 is the batch_size

    return tf.matmul(output, weights['out']) + biases['out']



with tf.variable_scope('forward-pass'):
    preds = RNN(x, weights, biases)

#define the cross entropy loss function
loss_fn = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=preds))
#define the optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=eta).minimize(loss_fn)


#calculate the number of correct predictions
correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(y, 1))
#calculate the average/mean accuracy
accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32))


#Create a handler to intialize all the variables
init = tf.global_variables_initializer()

#Train the network
with tf.Session() as sess:
    sess.run(init)
    step = 1

    #Keep training until max iterations is reached
    while (step*batch_size) < training_iters:
        #batch_x : (100, 784)   batch_y : (100, 10)
        batch_x, batch_y = mnist.train.next_batch(batch_size)

        #Reshape into (100,28,28) where for each sample we have 28 sequences of 28 length inputs
        batch_x = batch_x.reshape((batch_size, n_steps, n_input))

        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

        if step % display_step == 0:
            #calculate batch accuracy 
            acc = sess.run(accuracy, feed_dict={x:batch_x, y:batch_y})
            #calculate the loss
            loss = sess.run(loss_fn, feed_dict={x:batch_x, y:batch_y})

            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1

    print "Optimization Completed!"

    #Calculate accuracy for 1000 mnist images
    test_len = 1000
    test_x = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
    test_y = mnist.test.labels[:test_len]
    print ("Testing Accuracy : " + "{:.6f}".format(sess.run(accuracy, feed_dict={x: test_x, y:test_y})))





    



