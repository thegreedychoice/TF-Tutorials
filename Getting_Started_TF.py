import tensorflow as tf 
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

""" Tensor Constants """

print ("Tensor Constants Started! \n")
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
print(node1, node2)
sess = tf.Session()
print(sess.run([node1, node2]))
node3 = tf.add(node1, node2)
print("node3 : ", node3)
print("sess.run(node3) : ", sess.run(node3))
print ("Tensor Constants Ended! \n")


""" Tensor Placeholders """

print ("Tensor Placehlders Started! \n")

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b  #shortcut for tf.add(a,b)
add_multi_node = adder_node * 3
print("sess.run(adder_node, {a: 3, b:4.5})", sess.run(adder_node, {a: 3, b:4.5}))
print("sess.run(adder_node, {a: [1,3], b:[4,5]})", sess.run(adder_node, {a: [1,3], b:[4,5]}))
print("sess.run(add_multi_node, {a: 3, b: 5})", sess.run(add_multi_node, {a: 3, b: 5}))


print ("Tensor Placehlders Ended! \n")



""" Tensor Flow Train API to train a linear Regression Model """

print ("Tensor Flow Train Regression Model!")
TRAIN_METHOD = 3  #  1 for Core API, 2 for Predefinded Model from tf.contrib.learn, 3 for Custom Model

if TRAIN_METHOD == 1 :
	print("Training using TF Core API Started!")
	#Intialize  model paramteres and the model form 
	W = tf.Variable([0.3], tf.float32)
	b = tf.Variable([-0.3], tf.float32)
	x = tf.placeholder(tf.float32)
	y = tf.placeholder(tf.float32)
	linear_model = W * x + b
	#Initialize the sum squared loss function 
	loss_func = tf.reduce_sum(tf.square(linear_model - y))
	#Intialize the Gradient Descent Optimizer with learning rate 0.01
	optimizer = tf.train.GradientDescentOptimizer(0.01)
	train = optimizer.minimize(loss_func)
	#Initialize the training Data
	x_train = [1,2,3,4]
	y_train = [0,-1, -2, -3]
	#Intialize the Tensor flow variables W and b
	init = tf.global_variables_initializer()
	sess = tf.Session()
	sess.run(init)
	#Train the model
	epochs = 1000
	for i in range(epochs):
		sess.run(train, {x: x_train, y: y_train})

	#Evaluate the trained Model
	trained_W, trained_b, trained_loss = sess.run([W,b,loss_func], {x: x_train, y: y_train})
	print("Trained_W : %s  Trained_B : %s  Loss : %s "%(trained_W, trained_b, trained_loss))

	print("Training using TF Core API Ended!")


if TRAIN_METHOD == 2 :
	print("Training using TF Contrib Learn Pre-defined Model Started!")
	#Add all the features that are required to train. In this case only one real-valued feature
	#Can work with high dimensional complicated features as well 
	features = [tf.contrib.layers.real_valued_column("x", dimension=1)]
	#estimator respresents your model. In this case Linear Regression Model
	estimator = tf.contrib.learn.LinearRegressor(feature_columns=features)
	# Here we use `numpy_input_fn`. We have to tell the function how many batches
	# of data (num_epochs) we want and how big each batch should be.
	x = np.array([1., 2., 3., 4.])
	y = np.array([0., -1., -2., -3.])
	input_fn = tf.contrib.learn.io.numpy_input_fn({"x":x}, 
												  y, batch_size=4,
	                                              num_epochs=1000)

	# Train the Model
	estimator.fit(input_fn=input_fn, steps=1000)
	# Test the Model
	print(estimator.evaluate(input_fn=input_fn))

	print("Training using TF Contrib Learn Pre-defined Model Ended!")

if TRAIN_METHOD == 3 :
	print("Training by creating a Custom Model Started!")
	# Declare list of features, we only have one real-valued feature
	def model(features, labels, mode):
	  # Build a linear model and predict values
	  W = tf.get_variable("W", [1], dtype=tf.float64)
	  b = tf.get_variable("b", [1], dtype=tf.float64)
	  y = W*features['x'] + b
	  # Loss sub-graph
	  loss = tf.reduce_sum(tf.square(y - labels))
	  # Training sub-graph
	  global_step = tf.train.get_global_step()
	  optimizer = tf.train.GradientDescentOptimizer(0.01)
	  train = tf.group(optimizer.minimize(loss),
	                   tf.assign_add(global_step, 1))
	  # ModelFnOps connects subgraphs we built to the
	  # appropriate functionality.
	  return tf.contrib.learn.ModelFnOps(
	      mode=mode, predictions=y,
	      loss=loss,
	      train_op=train)

	estimator = tf.contrib.learn.Estimator(model_fn=model)
	# define our data set
	x = np.array([1., 2., 3., 4.])
	y = np.array([0., -1., -2., -3.])
	input_fn = tf.contrib.learn.io.numpy_input_fn({"x": x}, y, 4, num_epochs=1000)

	# train
	estimator.fit(input_fn=input_fn, steps=1000)
	# evaluate our model
	print(estimator.evaluate(input_fn=input_fn, steps=10))

	
	print("Training by creating a Custom Model Ended!")

