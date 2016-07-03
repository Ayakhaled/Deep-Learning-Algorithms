# Imports

import theano
import theano.tensor as T
import numpy as np
from six.moves import cPickle

#data set 
inputs = cPickle.load(open("train.pickle", "rb"))
#labels
Y = inputs[:, (0)] 
print Y.shape
#training set 
X = inputs[:, 1:]
#append ones "col"
X = np.append(np.ones((X.shape[0], 1)), X, axis = 1)
print X.shape 
#symbols

x = T.dmatrix('Inputs')
no_of_hidden_N= 25
no_of_features = X.shape[1]
no_of_classes = 10 
target = np.zeros((Y.shape[0],no_of_classes))

w1 = theano.shared(value = np.random.rand(no_of_features, no_of_hidden_N)* 0.01, name = 'Layer1 weights') 
w2 = theano.shared(value = np.random.rand(no_of_hidden_N, no_of_classes)* 0.01, name = 'Layer2 weights') 

h = T.dmatrix('w1.x') #Dimension: no_of_examples * no_of_featues(784)+1
activation_h = T.dmatrix('Layer2 outputs') 

#Final output
k = T.dmatrix('activation_h * w2') #Dimension: no_of_examples * no_of_classes
activation_k = T.dmatrix('Layer3 outputs')

t = T.dmatrix('Actual output')
delta_w1 = T.dmatrix('Delta w1')
delta_w2 = T.dmatrix('Delta w2')

eta = 0.1

#equations
h = T.dot(x, w1)
activation_h = T.nnet.sigmoid(h)

k = T.dot(activation_h, w2)
activation_k = T.nnet.softmax(k)

cost = (T.sum(T.sub(activation_k, t)**2)) / (2 * X.shape[0])

delta_w1 = T.grad(cost, w1)
delta_w2 = T.grad(cost, w2)

#w1 = w1 - eta * delta w
update_w1 = (w1, w1 - eta * delta_w1)
update_w2 = (w2, w2 - eta * delta_w2)

updates = [update_w1, update_w2]

for i in range (Y.shape[0]):
	value = Y[i]
	target[i][value] = value
	

train = theano.function([x, t], cost, updates = updates)

for i in range (100):
	print train(X, target) 



# Preprocess data



# Normalize data

# One hot encode input

# Declare theano symbols x, y, t, weights

# Declare theano equations (Netj, Activationj, Netk, Output, Gradients)

# Define theano function to train model and do updates

# Run loop to train