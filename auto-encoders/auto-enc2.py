import theano
import theano.tensor as T
import numpy as np
from six.moves import cPickle

#data set
#training  
X = cPickle.load(open("inputs1.pickle", "rb"))
#training set
X = (X - X.mean()) / X.std()

#variables
no_of_hidden_N= 100
no_of_features = X.shape[1]

cost_T = list()

#symbols

x = T.dmatrix('Inputs')
t = T.dmatrix('Actual output')

w1 = theano.shared(value = np.random.randn(no_of_features, no_of_hidden_N)* 0.01, name = 'Layer1 weights') 
w2 = theano.shared(value = np.random.randn(no_of_hidden_N, no_of_features)* 0.01, name = 'Layer2 weights') 

h = T.dmatrix('w1.x') #Dimension: no_of_examples * no_of_featues(784)+1
activation_h = T.dmatrix('Layer2 outputs') 

#Final output
k = T.dmatrix('activation_h * w2') #Dimension: no_of_examples * no_of_classes
activation_k = T.dmatrix('Layer3 outputs')

delta_w1 = T.dmatrix('Delta w1')
delta_w2 = T.dmatrix('Delta w2')

eta = 0.009

#equations
h = T.dot(x, w1)
# activation_h = T.nnet.sigmoid(h)
# activation_h = T.maximum(h, 0.01*h)
activation_h = T.tanh(h)

k = T.dot(activation_h, w2)
activation_k = k

cost = (T.sum(T.sub(activation_k, x)**2)) / (2 * x.shape[0])
#cost = T.nnet.binary_crossentropy(activation_k, t).mean()
#cost = T.mean(T.nnet.categorical_crossentropy(activation_k, t))

delta_w1 = T.grad(cost, w1)
delta_w2 = T.grad(cost, w2)

#w1 = w1 - eta * delta w
update_w1 = (w1, w1 - eta * delta_w1)
update_w2 = (w2, w2 - eta * delta_w2)

updates = [update_w1, update_w2]	


train = theano.function([x], cost, updates = updates)
activation1 = theano.function([x], activation_h)

for i in range (150):
	cost_T.append(train(X))
	print cost_T[i]


cPickle.dump(w1.eval(), open("weights2.pickle", "wb"), cPickle.HIGHEST_PROTOCOL)