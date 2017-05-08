import tensorflow as tf
import numpy as np 	
from tensorflow.examples.tutorials.mnist import input_data
import mnist

import time
import os.path
import argparse
import sys

def input_placeholders(batch_size):
	"""This function defines placeholders which contain information about the input shape for both images and labels, which is used by 
		the computational graph into which actual training examples are fed
	"""

	images_ph = tf.placeholder(tf.float32, shape=(batch_size, mnist.NUM_IMAGE_PIXELS))
	labels_ph = tf.placeholder(tf.int32, shape=(batch_size))
	return images_ph, labels_ph

def fill_feed_dictionary(data, images_ph, labels_ph):
	"""
	This method is used for updating the feed_dictionary with actual training data.
	It returns an object with feed_dict format, with the actual values from the placeholders
	conatained in them, to be used by the next step in training.
	Everytime this function is called, it updates the training data with the next batch of samples
	"""
	images_batch, labels_batch = data.next_batch(FLAGS.batch_size, FLAGS.fake_data)
	feed_dict = {
		images_ph : images_batch,
		labels_ph : labels_batch		
	}

	return feed_dict

def do_eval(sess,
            eval_correct,
            images_placeholder,
            labels_placeholder,
            data_set):
  """Runs one evaluation against all the examples in one full epoch and not just a selected training batch.
  """
  num_correct_preds = 0  
  steps_per_epoch = data_set.num_examples // FLAGS.batch_size
  num_examples = steps_per_epoch * FLAGS.batch_size
  for step in xrange(steps_per_epoch):
    feed_dict = fill_feed_dictionary(data_set,
                               images_placeholder,
                               labels_placeholder)
    num_correct_preds += sess.run(eval_correct, feed_dict=feed_dict)
  precision = float(num_correct_preds) / num_examples
  print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %(num_examples, num_correct_preds, precision))



def train_model():
	"""
	This method deals with training the model in mnist-model.py by running optimization over a number of steps.
	"""

	#read input da(ta first
	#read_data_sets() function will ensure that the correct data has been downloaded to your local training folder
	#and then unpack that data to return a dictionary of DataSet instances. 
	#FLAGS.fake_data can be ignored as it is used-for unit testing purposes
	print ("Training Started! ")
	data = input_data.read_data_sets(FLAGS.input_data_dir, FLAGS.fake_data)

	#We need to specify that our model will be used with the Default global Tensor Flow graph.
	#A default global TF graph tf.Graph() is a collection of ops that may be executed as a group
	with tf.Graph().as_default():
		# Generate placeholders for the images and labels.
	    images_ph, labels_ph = input_placeholders(FLAGS.batch_size)

	    # Build a Graph that computes predictions from the inference model.
	    logits = mnist.inference(images_ph, FLAGS.num_hidden1_nodes, FLAGS.num_hidden1_nodes)

	    # Add the loss calculation op to the default Graph
	    loss = mnist.loss(logits, labels_ph)

	    # Add the minimization op the Graph
	    train_op = mnist.training(loss, FLAGS.eta)

	    # Add the Evaluation to test predictions to the Graph
	    eval_correct = mnist.evaluation(logits, labels_ph)

	    # Build the summary Tensor based on the TF collection of Summaries.
	    summary = tf.summary.merge_all()

	    # Add the variable initializer Op. to the Graph
	    init = tf.global_variables_initializer()

	    #Create a Tensor Flow Saver for writing Training Checkpoints
	    saver = tf.train.Saver()

	    #Now once all the build preparation is completed and all the ops are added to the Graph,
	    #we need to create a Session in order to run the computaional Graph
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
	    	feed_dict = fill_feed_dictionary(data.train, images_ph, labels_ph)

	    	#Run one step of the training by running Ops train_op and loss
	    	#No need to store the activations returned by the train_op minimization step
	    	_, loss_val = sess.run([train_op, loss],
	    					    feed_dict=feed_dict)

	    	duration = time.time() - start_time

	    	#Record all the training summaries generates and print the training progress/statistics
	    	#after every 100th iteration
	    	if step%100 == 0:
	    		# Print status to stdout.
        		print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_val, duration))

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
                        data.train)
                print('Validation Data Evaluation:')
                do_eval(sess,
                        eval_correct,
                        images_ph,
                        labels_ph,
                        data.validation)
                # Evaluate against the test set.
                print('Test Data Evaluation:')
                do_eval(sess,
                        eval_correct,
                        images_ph,
                        labels_ph,
                        data.test)





def main(_):
	if tf.gfile.Exists(FLAGS.log_dir):
		print "MNIST Directory Overwritten!"
		tf.gfile.DeleteRecursively(FLAGS.log_dir)
  		tf.gfile.MakeDirs(FLAGS.log_dir)
  		train_model()
  	else:
  		print "MNIST Directory Created!"
   		tf.gfile.MakeDirs(FLAGS.log_dir)
        train_model()



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
	      default=2000,
	      help='Number of steps to run trainer.'
	  )
	  parser.add_argument(
	      '--num_hidden1_nodes',
	      type=int,
	      default=128,
	      help='Number of units in hidden layer 1.'
	  )
	  parser.add_argument(
	      '--num_hidden2_nodes',
	      type=int,
	      default=32,
	      help='Number of units in hidden layer 2.'
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
	  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)


	    		        








