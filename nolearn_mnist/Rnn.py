import numpy as np
from load_mnist import load_mnist_set
from nolearn.lasagne import NeuralNet
from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import Conv2DLayer
from lasagne.layers import MaxPool2DLayer
from lasagne.nonlinearities import softmax

X, y, X_test, y_test = load_mnist_set()

net = NeuralNet(
		layers = [
			('input',InputLayer),
			('hidden',DenseLayer),
			('output',DenseLayer),
		],

		input_shape = (None, X.shape[1], X.shape[2], X.shape[3]),	
		hidden_num_units = m,
		output_nonlinearity = softmax
		output_num_units = 10,

		update = nesterov_momentum,
		update_learning_rate = 0.01,
		update_momentum = 0.9,

		max_epochs = 100,
		verbose = 3,
)

net.fit(X,y)

y_pred1 = net1.predict(X_test_flat)
print "The accuracy of this network is: %0.5f" % (y_pred1 == y_test).mean()
