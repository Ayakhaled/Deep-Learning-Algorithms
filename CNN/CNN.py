import theano
import theano.tensor as T
import numpy as np
from six.moves import cPickle
import theano.tensor.signal.pool as pool 
#Data

#Training
inputs = cPickle.load(open("/home/aya/DeepLearning/train.pickle", "rb"))
Y = inputs[:, 0] #labels
X = inputs[:, 1:] #training set
X = (X - X.mean()) / X.std()

#Test 
inputs_test = cPickle.load(open("/home/aya/DeepLearning/test.pickle", "rb"))
Y_test = inputs_test[:, (0)]
X_test = inputs_test[:, 1:]
X_test = (X_test - X_test.mean()) / X_test.std()

X = X.reshape((-1, 1, 28, 28)).astype("float32")
X_test = X_test.reshape((-1, 1, 28, 28)).astype('float32')

#Variables
no_of_HN = 100
no_of_features = X.shape[1]
no_of_classes = 10 

eta = np.float32(0.05) #learning rate

target = np.zeros((Y.shape[0],no_of_classes)).astype('float32')
target_test = np.zeros((Y_test.shape[0], no_of_classes)).astype('float32')

cost_test = list()
cost_T = list()

#one hot
target[np.arange(Y.shape[0]), Y] = 1
target_test[np.arange(Y_test.shape[0]), Y_test] = 1

#symbols
images = T.ftensor4(name='images')
t = T.fmatrix('Actual output').astype('float32')

FILTER = theano.shared(value=np.random.randn(1, 1, 3, 3).astype('float32'), name="Filter")
wi = theano.shared(value=np.random.randn(14*14, no_of_HN).astype('float32') * 0.01, name="Wi") #
wj = theano.shared(value=np.random.randn(no_of_HN, no_of_classes).astype('float32') * 0.01, name="Wj")

conv = T.nnet.conv2d(images, FILTER, border_mode = (1, 1)) #convolution and padding 
activation_conv = T.nnet.relu(conv) 

pool_out = pool.pool_2d(activation_conv, (2, 2), ignore_border=False) 
flatten = pool_out.flatten(2)

Neti = T.dot(flatten, wi)
ai = T.nnet.relu(Neti)

Netj = T.dot(ai, wj)
y = T.nnet.softmax(Netj)

cost = T.mean(T.nnet.categorical_crossentropy(y, t))

delta_wi = T.grad(cost, wi)
delta_wj = T.grad(cost, wj)
delta_FILTER = T.grad(cost, FILTER)

update_wi = (wi, wi - eta * delta_wi)
update_wj = (wj, wj - eta * delta_wj)
update_FILTER = (FILTER, FILTER - eta * delta_FILTER)

updates = [update_wi, update_wj, update_FILTER]

train = theano.function([images, t], cost, updates = updates)
validate = theano.function([images, t], cost)
forwardProp = theano.function([images], y)

for i in range(1500):
	cost_T.append(train(X, target))
	print cost_T[i]
	if(i % 10 == 0):
		results = np.argmax(forwardProp(X_test), axis=1) #gets the index of the maximum element in output "after activation"
		count = len(np.where(Y_test == results)[0])

		accuracy = float(count) / float(X_test.shape[0])

		print "Accuracy at epoch: " + str(i) + ": %.2f" % float(accuracy * 100) + "%"
