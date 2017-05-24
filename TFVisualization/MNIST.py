from Networks import *
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.examples.tutorials.mnist import input_data

import time
import os.path
import argparse
import sys

IMAGE_HEIGHT = 28
IMAGE_WIDTH = 28
NUM_IMAGE_PIXELS = IMAGE_HEIGHT * IMAGE_WIDTH
NUM_CLASSES = 10 #0-9
import time
import os.path
import argparse
import sys

"""
This file will build a MNIST Model to recognize handwritten digits by training and later evaluating the model.
Main functions :

inference() : It builds the model
loss() : calculates the loss of the model at a particular time.
evaluate() : evaluate the model
"""

def activation_summary(x):
    """
    :param
        x: The activation value
    :return:
        None
    """
    #Record the histogram summary of activations
    tf.summary.histogram(x.op.name+'/activations', x)
    #Record the summary of sparsity of activations
    #tf.summary.scalar(x.op.name+'/sparsity', x)


def inference(input, keep_prob):
    """
    :param
        input: The input dataset of mnist images
    :return:
        logits : The predicted values for the entire dataset
    """
    #Reshape the input dataset before applying convolution
    images = tf.reshape(input, [-1, 28, 28, 1])

    with tf.name_scope('conv_layer1') as scope:
        net_conv1 = net_convolution2d(images, 32, (5, 5, 1), [1, 1, 1, 1], 'SAME')
        act_conv1 = reLu(net_conv1)
        activation_summary(act_conv1)
        out_pool1 = max_pool2d(act_conv1, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME', name='pool_layer1')

    with tf.variable_scope('conv_layer2') as scope:
        net_conv2 = net_convolution2d(out_pool1, 64, (5, 5, 32), [1, 1, 1, 1], 'SAME')
        act_conv2 = reLu(net_conv2, name='conv2')
        activation_summary(act_conv2)
        out_pool2 = max_pool2d(act_conv2, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME', name='pool_layer2')

	hidden_nodes = 1024
    #the output size after second convolution is 7 by 7 by 64
    with tf.variable_scope('fcn_layer1') as scope:
        w_fcn1 = initialize_weights([7*7*64, hidden_nodes])
        b_fcn1 = initialize_biases([hidden_nodes])
        #Reshape or flatten the out_pool2 in order to send it to the fully connected layer
        inp_fcn1 = tf.reshape(out_pool2, [-1, 7*7*64])
        net_fcn1 = tf.matmul(inp_fcn1, w_fcn1) + b_fcn1
        act_fcn1 = reLu(net_fcn1, name='fcn1')
        activation_summary(act_fcn1)

    with tf.variable_scope('dropout_layer1') as scope:
        out_drop1 = dropout(input=act_fcn1, keep_prob=keep_prob, name=scope.name)

    # Fully Connected/Readout Layer
    # This gives the final output layer of un-normalized probabilities for each label
    with tf.variable_scope('readout_layer') as scope:
        w_fcn2 = initialize_weights([hidden_nodes, 10])
        b_fcn2 = initialize_biases([10])
        logits = tf.matmul(out_drop1, w_fcn2) + b_fcn2
        activation_summary(logits)
    return logits

def losses(logits, labels):
	"""
	This functions calculates the average cross entropy loss for the given set of input
	Add summary for "Loss" and "Loss/avg".
  	Args:
    	logits: Logits from inference().
    	labels: Labels from inputs(). 1-D tensor
        	    of shape [batch_size]
  	Returns:
    	Loss tensor of type float.

	"""
  	cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits(
        labels=labels,logits=logits, name='cross_entropy_per_example')
  	cross_entropy_mean = tf.reduce_mean(cross_entropy_loss, name='cross_entropy_mean')
  	return cross_entropy_mean


def training(loss, eta):
	""" This method futher adds ops to the graph to minimize the loss by optimization using gradient descent.
	"""
	#this op basically generates summary values and emits snapshot value of loss everytime
    #the summary is written out
	tf.summary.scalar('loss', loss)
	#optimizer = tf.train.GradientDescentOptimizer(eta)
	optimizer = tf.train.AdamOptimizer(1e-4)
	#this variable holds a counter with an initial value 0, to be used as a counter in training
	global_counter = tf.Variable(0, name='global_counter', trainable=False)
	train_op = optimizer.minimize(loss, global_step=global_counter)
	return train_op

def evaluation(o, y):
    # Evaluate Predictions and Accuracy
    correct_predictions = tf.equal(tf.argmax(y, 1), tf.argmax(o, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    return accuracy




def input_placeholders():
    images_ph = tf.placeholder(tf.float32, shape=[None, 784])
    labels_ph = tf.placeholder(tf.int32, shape=[None, 10])
    keep_prob_ph = tf.placeholder(tf.float32)
    return images_ph, labels_ph, keep_prob_ph

def fill_feed_dictionary(data, images_ph, labels_ph, keep_prob, eval=False):
    """
    	This method is used for updating the feed_dictionary with actual training data.
    	It returns an object with feed_dict format, with the actual values from the placeholders
    	conatained in them, to be used by the next step in training.
    	Everytime this function is called, it updates the training data with the next batch of samples
    """
    keep_probability = 0.5
    if eval:
        keep_probability = 0.1
    images_batch, labels_batch = data.next_batch(FLAGS.batch_size, FLAGS.fake_data)

    feed_dict = {
        images_ph: images_batch,
        labels_ph: labels_batch,
        keep_prob: keep_probability
    }

    return feed_dict



def do_eval(sess,
            eval_correct,
            images_placeholder,
            labels_placeholder,
            keep_prob_placeholder,
            data_set):
  """Runs one evaluation against all the examples in one full epoch and not just a selected training batch.
  """
  num_correct_preds = 0
  steps_per_epoch = data_set.num_examples // FLAGS.batch_size
  num_examples = steps_per_epoch * FLAGS.batch_size
  for step in xrange(steps_per_epoch):
    feed_dict = fill_feed_dictionary(data_set,
                               images_placeholder,
                               labels_placeholder, keep_prob_placeholder, eval=True)
    num_correct_preds += sess.run(eval_correct, feed_dict=feed_dict)
  precision = float(num_correct_preds) / num_examples
  print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %(num_examples, num_correct_preds, precision))



def train_interactive():
	"""
        This method deals with training the model in mnist-model.py by running optimization over a number of steps.
	"""
	mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

	# Build the computational graph and create an interactive session
	# if not using Interactive Session, the entire computational graph should be built before launching graph
	sess = tf.InteractiveSession()

	# Intialize the placeholder values for training data
	x, y, keep_prob = input_placeholders()


	o = inference(x, keep_prob=0.5)
	loss = losses(logits=o, labels=y)

	eta = 1e-4

	train_op = training(loss, eta)
	accuracy = evaluation(o, y)

	# Build the summary Tensor based on the TF collection of Summaries.
	summary = tf.summary.merge_all()

	# We also need to create a TF Summary Writer in order to record all the summaries and the Graph
	summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

	# Intialize the Global Variables to zeros
	sess.run(tf.global_variables_initializer())


	epochs =  300
	for i in range(epochs):
		batch = mnist.train.next_batch(50)
		feed_dict = {x: batch[0], y: batch[1], keep_prob: 0.5}
		if i %100 == 0:
			feed_dict = {x: batch[0], y: batch[1], keep_prob: 1.0}
			train_accuracy = accuracy.eval(feed_dict=feed_dict)
			print("Epoch No : %d  Training Accuracy : %g "%(i, train_accuracy))
		train_op.run(feed_dict=feed_dict)
		summary_str = sess.run(summary, feed_dict=feed_dict)
		summary_writer.add_summary(summary_str, i)
		summary_writer.flush()
	#Test your Model With Test Data
	print("Testing the Model :")
	print("Test Accuracy : %g "%(accuracy.eval(feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0 })))






def train_model():
	"""
	This method deals with training the model in mnist-model.py by running optimization over a number of steps.
	"""

	#read input da(ta first
	#read_data_sets() function will ensure that the correct data has been downloaded to your local training folder
	#and then unpack that data to return a dictionary of DataSet instances.
	#FLAGS.fake_data can be ignored as it is used-for unit testing purposes
	print ("Training Started! ")
	data = input_data.read_data_sets(FLAGS.input_data_dir, FLAGS.fake_data, one_hot=True)

	#We need to specify that our model will be used with the Default global Tensor Flow graph.
	#A default global TF graph tf.Graph() is a collection of ops that may be executed as a group
	with tf.Graph().as_default():
		# Generate placeholders for the images and labels.
	    images_ph, labels_ph, keep_prob_ph = input_placeholders()

	    #Build a Graph that computes predictions from the inference model.
	    logits = inference(images_ph, keep_prob_ph)

	    # Add the loss calculation op to the default Graph
	    loss = losses(logits, labels_ph)

	    # Add the minimization op the Graph
	    train_op = training(loss, FLAGS.eta)

	    # Add the Evaluation to test predictions to the Graph
	    eval_correct = evaluation(logits, labels_ph)

	    # Build the summary Tensor based on the TF collection of Summaries.
	    summary = tf.summary.merge_all()

	    # Add the variable initializer Op. to the Graph
	    init = tf.global_variables_initializer()

	    #Create a Tensor Flow Saver for writing Training Checkpoints
	    saver = tf.train.Saver()

	    #Now once all the build preparation is completed and all the ops are added to the Graph,
	    #we need to create a Session in order to run the computational Graph
	    sess = tf.Session()

	    #We also need to create a TF Summary Writer in order to record all the summaries and the Graph
	    summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

	    #Now all the required Ops are attached to the Default Graph and all is built,
	    #Start the session by initialing all the TF variables.
	    sess.run(init)

	    #Start the Training
	    for step in xrange(FLAGS.max_steps):
	    	start_time = time.time()

	    	#Update the feed_dict with the next batch of samples to train  with.
	    	feed_dict = fill_feed_dictionary(data.train, images_ph, labels_ph, keep_prob_ph)

	    	#Run one step of the training by running Ops train_op and loss
	    	#No need to store the activations returned by the train_op minimization step
	    	_, loss_val = sess.run([train_op, loss],
	    					    feed_dict=feed_dict)

	    	duration = time.time() - start_time

	    	#Record all the training summaries generates and print the training progress/statistics
	    	#after every 100th iteration
	    	if step%100 == 0:
	    		# Print status to stdout.
        		print('Step %d: loss = %.2f (%.3f sec)' % (step, 0.1, duration))

        		#Now we need to update the events file with summaries
        		#Run the summary Op attached to the Graph
        		#Everytime the summary is evaluated, new summaries are written into the events files
        		summary_str = sess.run(summary, feed_dict=feed_dict)
        		summary_writer.add_summary(summary_str, step)
        		summary_writer.flush()

            #Save the Model at every 1000th iteration and perform evaluation on complete data
        	if (step + 1) % 1000 == 0 or (step + 1) == FLAGS.max_steps:
        		checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
        		saver.save(sess, checkpoint_file, global_step=step)
                #Evaluate against the training set
                print('Training Data Evaluation:')
                do_eval(sess,
                        eval_correct,
                        images_ph,
                        labels_ph,
                        keep_prob_ph,
                        data.train)
                print('Validation Data Evaluation:')
                do_eval(sess,
                        eval_correct,
                        images_ph,
                        labels_ph,
                        keep_prob_ph,
                        data.validation)
                # Evaluate against the test set.
                print('Test Data Evaluation:')
                do_eval(sess,
                        eval_correct,
                        images_ph,
                        labels_ph,
                        keep_prob_ph,
                        data.test)





def main():

	if tf.gfile.Exists(FLAGS.log_dir):
		print "MNIST Directory Overwritten!"
		tf.gfile.DeleteRecursively(FLAGS.log_dir)
  		tf.gfile.MakeDirs(FLAGS.log_dir)
  		train_interactive()
  	else:
  		print "MNIST Directory Created!"
   		tf.gfile.MakeDirs(FLAGS.log_dir)
        train_interactive()
	return

	#train()



if __name__ == '__main__':
	  parser = argparse.ArgumentParser()
	  parser.add_argument(
	      '--eta',
	      type=float,
	      default=0.01,
	      help='Initial learning rate.'
	  )
	  parser.add_argument(
	      '--max_steps',
	      type=int,
	      default=1000,
	      help='Number of steps to run trainer.'
	  )

	  parser.add_argument(
	      '--batch_size',
	      type=int,
	      default=100,
	      help='Batch size.  Must divide evenly into the dataset sizes.'
	  )
	  parser.add_argument(
	      '--input_data_dir',
	      type=str,
	      default=os.path.dirname(os.path.abspath(__file__))+'/tmp/mnist/input_data',
	      help='Directory to put the input data.'
	  )
	  parser.add_argument(
	      '--log_dir',
	      type=str,
	      default=os.path.dirname(os.path.abspath(__file__))+'/tmp/mnist/logs/fully_connected_feed',
	      help='Directory to put the log data.'
	  )
	  parser.add_argument(
	      '--fake_data',
	      default=False,
	      help='If true, uses fake data for unit testing.',
	      action='store_true'
	  )

	  FLAGS, unparsed = parser.parse_known_args()
	  print FLAGS.batch_size
	  #tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
	  main()





