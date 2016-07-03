import theano
import theano.tensor as T
import numpy as np
from six.moves import cPickle

#--data set
weights1 = cPickle.load(open("weights1.pickle", "rb"))
weights2 = cPickle.load(open("weights2.pickle", "rb"))

#-training set 
inputs = cPickle.load(open("train.pickle", "rb"))
#labels
Y = inputs[:, 0] 
#training set 
X = inputs[:, 1:]
X = (X - X.mean()) / X.std()
#append ones "col"
X = np.append(np.ones((X.shape[0], 1)), X, axis = 1)

#-test set
inputs_test = cPickle.load(open("test.pickle", "rb"))
Y_test = inputs_test[:, (0)]
X_test = inputs_test[:, 1:]
X_test = (X_test - X_test.mean()) / X_test.std()
X_test = np.append(np.ones((X_test.shape[0], 1)), X_test, axis = 1)

#--variables
no_of_hidden_NL1= 300
no_of_hidden_NL2= 100

no_of_features = X.shape[1]
no_of_classes = 10 

target = np.zeros((Y.shape[0],no_of_classes))
target_v = np.zeros((Y_test.shape[0], no_of_classes))

cost_V = list()
cost_T = list()

target[np.arange(Y.shape[0]), Y] = 1
target_v[np.arange(Y_test.shape[0]), Y_test] = 1

#--symbols

x = T.dmatrix('Inputs')
t = T.dmatrix('Actual output')

w1 = theano.shared(value = weights1, name = 'Layer1 weights') 
w2 = theano.shared(value = weights2, name = 'Layer2 weights') 
w3 = theano.shared(value = np.random.rand(no_of_hidden_NL2, no_of_classes)* 0.1, name = 'Layer3 weights') 

h = T.dmatrix('w1.x') #Dimension: no_of_examples * no_of_featues(784)+1
activation_h = T.dmatrix('Layer2 outputs') 

i = T.dmatrix('activation_h * w2') #Dimension: no_of_examples * no_of_featues(784)+1
activation_i = T.dmatrix('Layer3 outputs') 


#Final output
k = T.dmatrix('activation_i * w3') #Dimension: no_of_examples * no_of_classes
activation_k = T.dmatrix('Layer4 outputs')

delta_w1 = T.dmatrix('Delta w1')
delta_w2 = T.dmatrix('Delta w2')
delta_w3 = T.dmatrix('Delta w3')

eta = 0.4

#equations
h = T.dot(x, w1)
# activation_h = T.nnet.sigmoid(h)
# activation_h = T.maximum(h, 0.01*h)
activation_h = T.tanh(h)

i = T.dot(activation_h, w2)
activation_i = T.tanh(i)

k = T.dot(activation_i, w3)
activation_k = T.nnet.softmax(k)

#cost = (T.sum(T.sub(activation_k, t)**2)) / (2 * X.shape[0])
#cost = T.nnet.binary_crossentropy(activation_k, t).mean()
cost = T.mean(T.nnet.categorical_crossentropy(activation_k, t))

delta_w1 = T.grad(cost, w1)
delta_w2 = T.grad(cost, w2)
delta_w3 = T.grad(cost, w3)

#w1 = w1 - eta * delta w
update_w1 = (w1, w1 - eta * delta_w1)
update_w2 = (w2, w2 - eta * delta_w2)
update_w3 = (w3, w3 - eta * delta_w3)

updates = [update_w1, update_w2, update_w3]	


train = theano.function([x, t], cost, updates = updates)
validate = theano.function([x, t], cost)

forwardProp = theano.function([x], activation_k)

for i in range (1500):
	# print train(X, target) 
	#print validate(X_test, target_v)
	cost_T.append(train(X, target))
	print cost_T[i]
	cost_V.append(validate(X_test, target_v))

	if(i % 10 == 0):

		results = np.argmax(forwardProp(X_test), axis = 1)
		count = len(np.where(Y_test == results)[0])

		#for i in range(results.shape[0]):
		#	index = results[i]
		#	if (target_v[i][index] == 1):
		#		count += 1

		accuracy = float(count) / float(X_test.shape[0])
		print 'Accuracy at epoch ' + str(i) + ': %.2f' % float(accuracy * 100) + '%'


# Preprocess data

# Normalize data

# One hot encode input

# Declare theano symbols x, y, t, weights

# Declare theano equations (Netj, Activationj, Netk, Output, Gradients)

# Define theano function to train model and do updates

# Run loop to train