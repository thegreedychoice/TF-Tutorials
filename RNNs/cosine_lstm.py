import numpy as np
import tensorflow as tf 
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


dataset = np.cos(np.arange(1000)*(20*np.pi/1000))[:,None]
plt.plot(dataset)

#convert the array of values into matrix form with appropiate X & Y values
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back):
		dataX.append(dataset[i:(i+look_back), 0])
		dataY.append(dataset[i+look_back, 0])

	return np.array(dataX), np.array(dataY)

#Window of 20 timesteps
look_back = 20
scaler = MinMaxScaler(feature_range=(0, 1))

#Scale the dataset in the range (0,1)
dataset = scaler.fit_transform(dataset)

#Create the training and test dataset
train_size = int(len(dataset) * 0.67)


train_set = dataset[0:train_size,:]
test_set = dataset[train_size:len(dataset),:] 

trainX, trainY = create_dataset(train_set, look_back)
test_X, test_Y = create_dataset(test_set, look_back)

trainX = tf.reshape(trainX, [trainX.shape[0], trainX.shape[1]])
test_X = tf.reshape(test_X, [test_X.shape[0], test_X.shape[1]])


batch_size = 1
LSTM_CELL_SIZE = look_back #Number of timesteps 


lstm_cell = tf.contrib.rnn.BasicLSTMCell(LSTM_CELL_SIZE, state_is_tuple=True)
state = (tf.zeros([trainX.shape[0], LSTM_CELL_SIZE]),)*2 



with tf.variable_scope('lstm_cosine') as scope:
	output, state_new = lstm_cell(trainX, state)


#Intialize the placeholder values for training data
x = tf.placeholder(tf.float64, shape = trainX.shape) #Each mnist image is 28 by 28 pixel image
y = tf.placeholder(tf.float64, shape = trainY.shape) # Each label is represented one-hot encode format 10-D(zero through nine)

sess = tf.InteractiveSession()


#Training the Network

#define Cross Entropy as Loss Function
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output[:,-1]))
#Use ADAM optimizer
optimizer = tf.train.AdamOptimizer(1e-4)
train_step = optimizer.minimize(loss)


sess.run(tf.global_variables_initializer())

#Optimization over epochs
#Logging over over every 100th iteration to evaluate the model
epochs = 20000
train_data = [trainX, trainY]
for i in range(epochs):
	batch = train_data.next_batch(batch_size)
	train_step.run(feed_dict={x: batch[0], y: batch[1]})
	print loss.eval()





