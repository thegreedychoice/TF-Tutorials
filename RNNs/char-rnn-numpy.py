import numpy as np

#Read the input file
data = open('data/input-char.txt', 'r').read()

#Get a list of all the unique characters from the input file
#i.e, generate a vocabulary
vocab = list(set(data))

data_size , vocab_size = len(data), len(vocab)

#Generate dictionaries to index the vocabulary
#They will be used for encoding/decoding the input later
char_to_idx = {ch:i for i, ch in enumerate(vocab)}
idx_to_char = {i:ch for i,ch in enumerate(vocab)}

print 'The input file has {0} characters and {1} unique characters'.format(data_size, vocab_size)


#Hyperparameters
hidden_size = 100
seq_length = 25
eta = 1e-1

#Define the parameters
Wxh = np.random.randn(hidden_size, vocab_size)*0.01 #input to hidden
Whh = np.random.randn(hidden_size, hidden_size)*0.01 #hidden to hidden
Why = np.random.randn(vocab_size, hidden_size)*0.01 #vocab to hidden
Bh = np.zeros((hidden_size,1))   #hidden bias
By = np.zeros((vocab_size, 1))   #output bias


def compute_loss(inputs, targets, hprev):
	"""
	Args -
	inputs : List of input in the form of integars [5,21,9,46...] for all timesteps
	targets : List of targets in the form of integars [23,8,1,0...] for all timesteps
	hprev : Initial Hidden State of the shape (hidden_size, 1)

	Returns -
	loss : the total loss for the sequence
	gradients : model paramters gradients
	hs : final hidden state

	"""
	loss = 0
	hs, ys, ps, xs = {}, {}, {}, {}
	hs[-1] = np.copy(hprev)

    #forward pass
    #The equations are :
    #h(t) = tanh(Wxh*x(t) + Whh*h(t-1) + By)
    #y(t) = Why.h(t) + By
    #p(t) = softmax(y(t))
    #Loss += -np.log(ps(t)[correct class])
	for t in range(len(inputs)):
		#generate the one hot encoding vector for inputs
		xs[t] = np.zeros((vocab_size, 1))
		xs[t][inputs[t]] = 1

		#compute the hidden state at each timestep
		hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + Bh)
		#compute the output/un-normalized log probabilities at each timestep
		#This output predicts which character would come next after x[t]
		ys[t] = np.dot(Why, hs[t]) + By
		#compute the softmax/normalized probabilities of the next character
		ps[t] = np.exp(ys[t])/np.sum(np.exp(ys[t]))
		#calculate the negative log likelihood/cross entropy loss
		loss += -np.log(ps[t][targets[t], 0])


	#backward pass : BPTT

	#initialize the paramters gradients
	dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
	dBh, dBy = np.zeros_like(Bh), np.zeros_like(By)
    #gradient wrt to the next hidden state
	dhnext = np.zeros_like(hs[0])
	for t in reversed(xrange(len(inputs))):
	    dy = np.copy(ps[t])
	    dy[targets[t]] -= 1 # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
	    dWhy += np.dot(dy, hs[t].T)
	    dBy += dy
	    dh = np.dot(Why.T, dy) + dhnext # backprop into h
	    dhraw = (1 - hs[t] * hs[t]) * dh # backprop through tanh nonlinearity
	    dBh += dhraw
	    dWxh += np.dot(dhraw, xs[t].T)
	    dWhh += np.dot(dhraw, hs[t-1].T)
	    dhnext = np.dot(Whh.T, dhraw)
	for dparam in [dWxh, dWhh, dWhy, dBh, dBy]:
	    np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients

	return loss, dWxh, dWhh, dWhy, dBh, dBy, hs[len(inputs)-1]	    





def sample(hs, seed_ix, n):
	"""
	Sample a sequence of integars from the model by generating a probability distribution 
	over all the characters/integars after each timestep
	Args -
	h : is the hidden state 
	seed_ix : seed integar at timestep 0
	n : no of samples to be generated
	"""

	xs = np.zeros((vocab_size, 1))
	xs[seed_ix] = 1
	ixes = []

	for t in range(n):
		hs = np.tanh(np.dot(Wxh, xs) + np.dot(Whh, hs) + Bh)
		ys = np.dot(Why, hs) + By
		ps = np.exp(ys)/np.sum(np.exp(ys))
		ix = np.random.choice(range(vocab_size), p=ps.ravel())
		xs = np.zeros((vocab_size, 1))
		xs[ix] = 1
		ixes.append(ix)
	return ixes


#Do the training


n, p = 0, 0
mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mBh, mBy = np.zeros_like(Bh), np.zeros_like(By) # memory variables for Adagrad
smooth_loss = -np.log(1.0/vocab_size)*seq_length # loss at iteration 0
while True:
  # prepare inputs (we're sweeping from left to right in steps seq_length long)
  if p+seq_length+1 >= len(data) or n == 0: 
    hprev = np.zeros((hidden_size,1)) # reset RNN memory
    p = 0 # go from start of data
  inputs = [char_to_idx[ch] for ch in data[p:p+seq_length]]
  targets = [char_to_idx[ch] for ch in data[p+1:p+seq_length+1]]

  # sample from the model now and then
  if n % 100 == 0:
    sample_ix = sample(hprev, inputs[0], 200)
    txt = ''.join(idx_to_char[ix] for ix in sample_ix)
    #print '----\n %s \n----' % (txt, )


  # forward seq_length characters through the net and fetch gradient
  loss, dWxh, dWhh, dWhy, dBh, dBy, hprev = compute_loss(inputs, targets, hprev)
  smooth_loss = smooth_loss * 0.999 + loss * 0.001
  if n % 100 == 0: print 'iter %d, loss: %f' % (n, smooth_loss) # print progress
  
  # perform parameter update with Adagrad
  for param, dparam, mem in zip([Wxh, Whh, Why, Bh, By], 
                                [dWxh, dWhh, dWhy, dBh, dBy], 
                                [mWxh, mWhh, mWhy, mBh, mBy]):
    mem += dparam * dparam
    param += -eta * dparam / np.sqrt(mem + 1e-8) # adagrad update

  p += seq_length # move data pointer
  n += 1 # iteration counter 









