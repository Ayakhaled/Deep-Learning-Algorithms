import theano 
import theano.tensor as T
import numpy as np

#symbols
x = T.dmatrix('Inputs')
y = T.dmatrix('Predicted outputs')
t = T.dmatrix('Actual outputs')
delta_w = T.dmatrix('Delta w')

eta = 0.05

w = theano.shared(value = np.zeros((1, 3), dtype = theano.config.floatX), name = 'weights')

#Equations
y = T.dot(x, w.transpose())
cost = (T.sum(T.sub(y, t)**2)) / (2 * y.shape[0])


delta_w = T.grad(cost, w)

update_w = w - eta * delta_w
updates = [(w, update_w)]

#functions 
train = theano.function([x, t], cost, updates = updates)

#numpy variables 


# Load Data
inputs = np.load("x.npy")
results = np.load("t.npy").T


for i in range(500):
	print(train(inputs, results))
	
#print(w.eval())
