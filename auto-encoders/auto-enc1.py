import theano
import theano.tensor as T
import numpy as np
from six.moves import cPickle

#data set
#training  
inputs = cPickle.load(open("train.pickle", "rb"))
#labels
Y = inputs[:, 0] 
#training set 
X = inputs[:, 1:]
X = (X - X.mean()) / X.std()
#append ones "col"
X = np.append(np.ones((X.shape[0], 1)), X, axis = 1)

#test 
inputs_test = cPickle.load(open("test.pickle", "rb"))
Y_test = inputs_test[:, (0)]
X_test = inputs_test[:, 1:]
X_test = (X_test - X_test.mean()) / X_test.std()
X_test = np.append(np.ones((X_test.shape[0], 1)), X_test, axis = 1)

#variables
no_of_hidden_N= 300
no_of_features = X.shape[1]
no_of_classes = 10 

target = np.zeros((Y.shape[0],no_of_classes))
target_v = np.zeros((Y_test.shape[0], no_of_classes))

cost_V = list()
cost_T = list()

target[np.arange(Y.shape[0]), Y] = 1
target_v[np.arange(Y_test.shape[0]), Y_test] = 1

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
validate = theano.function([x], cost)

forwardProp = theano.function([x], activation_k)
activation1 = theano.function([x], activation_h)

for i in range (150):
	# print train(X, target) 
	#print validate(X_test, target_v)
	cost_T.append(train(X))
	print cost_T[i]
	cost_V.append(validate(X_test))


cPickle.dump(w1.eval(), open("weights1.pickle", "wb"), cPickle.HIGHEST_PROTOCOL)
inputs1 = activation1(X)
cPickle.dump(inputs1, open("inputs1.pickle", "wb"), cPickle.HIGHEST_PROTOCOL)
