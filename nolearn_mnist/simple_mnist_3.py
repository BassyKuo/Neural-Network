#!/usr/bin/env python
# Filename: simple_mnist.py
#Source: http://nbviewer.jupyter.org/github/dnouri/nolearn/blob/master/docs/notebooks/CNN_tutorial.ipynb
import matplotlib.pyplot as plt
import numpy as np
from load_mnist import load_mnist_set

from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import Conv2DLayer
from lasagne.layers import MaxPool2DLayer
from lasagne.nonlinearities import softmax
from lasagne.updates import adam
from lasagne.layers import get_all_params

from nolearn.lasagne import NeuralNet
from nolearn.lasagne import TrainSplit
from nolearn.lasagne import objective
from nolearn.lasagne import PrintLayerInfo

X, y, X_test, y_test = load_mnist_set()

## here will print the label and image
#figs, axes = plt.subplots(4, 4, figsize=(6, 6))
#for i in range(4):
#	for j in range(4):
#		axes[i, j].imshow(-X[i + 4 * j].reshape(28, 28), 
#						  cmap='gray',
#						  interpolation='none')
#		axes[i ,j].set_xticks([])
#		axes[i, j].set_yticks([])
#		axes[i, j].set_title("Label: {}".format(y[i + 4 * j]))
#		axes[i, j].axis('off')

# Here is an architecture with too mach maxpooling. For illustrative purposes, we set the pad parameter to 1; without it, the image size would shrink below 0, at which point the code will raise an error.

layers3 = [
    (InputLayer, {'shape': (None, X.shape[1], X.shape[2], X.shape[3])}),

	(Conv2DLayer, {'num_filters': 32, 'filter_size': (3, 3), 'pad': 1}),
	(Conv2DLayer, {'num_filters': 32, 'filter_size': (3, 3), 'pad': 1}),
	(MaxPool2DLayer, {'pool_size': (2, 2)}),
	(Conv2DLayer, {'num_filters': 32, 'filter_size': (3, 3), 'pad': 1}),
	(Conv2DLayer, {'num_filters': 32, 'filter_size': (3, 3), 'pad': 1}),
	(MaxPool2DLayer, {'pool_size': (2, 2)}),
	(Conv2DLayer, {'num_filters': 64, 'filter_size': (3, 3), 'pad': 1}),
	(Conv2DLayer, {'num_filters': 64, 'filter_size': (3, 3), 'pad': 1}),
	(MaxPool2DLayer, {'pool_size': (2, 2)}),
	(Conv2DLayer, {'num_filters': 64, 'filter_size': (3, 3), 'pad': 1}),
	(Conv2DLayer, {'num_filters': 64, 'filter_size': (3, 3), 'pad': 1}),
	(MaxPool2DLayer, {'pool_size': (2, 2)}),

	(DenseLayer, {'num_units': 64}),
	(DropoutLayer, {}),
	(DenseLayer, {'num_units': 64}),

	(DenseLayer, {'num_units': 10, 'nonlinearity': softmax}),
]
net3 = NeuralNet(
	layers=layers3,
	max_epochs=10,
	update_learning_rate=0.01,
	verbose=2,
)

# Show more information
net3.initialize()
layer_info = PrintLayerInfo()
layer_info(net3)

# train the net 
net3.fit(X, y)

# test
print "Start to test....."
y_pred = net3.predict(X_test)
print "The accuracy of this network is: %0.2f" % (y_pred == y_test).mean()

# store the network module
import cPickle as pickle
with open('results/simple_net3.pickle','wb') as f:
	pickle.dump(net3, f, -1)
