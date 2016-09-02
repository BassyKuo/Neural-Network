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

# here will print the label and image
figs, axes = plt.subplots(4, 4, figsize=(6, 6))
for i in range(4):
	for j in range(4):
		axes[i, j].imshow(-X[i + 4 * j].reshape(28, 28), 
						  cmap='gray',
						  interpolation='none')
		axes[i ,j].set_xticks([])	# x-axis scale
		axes[i, j].set_yticks([])	# y-axis scale
		axes[i, j].set_title("Label: {}".format(y[i + 4 * j]))
		axes[i, j].axis('off')	# hide the axis line and axis value
plt.show()

layers1 = [
	(InputLayer, {'shape': (None, X.shape[1], X.shape[2], X.shape[3])}),
	(Conv2DLayer, {'num_filters': 32, 'filter_size': (3, 3)}),
	(MaxPool2DLayer, {'pool_size': (2, 2)}),
	(Conv2DLayer, {'num_filters': 64, 'filter_size': (3, 3)}),
	(Conv2DLayer, {'num_filters': 64, 'filter_size': (3, 3)}),
	(MaxPool2DLayer, {'pool_size': (2, 2)}),
	(Conv2DLayer, {'num_filters': 96, 'filter_size': (3, 3)}),
	(MaxPool2DLayer, {'pool_size': (2, 2)}),
	(DenseLayer, {'num_units': 64}),
	(DropoutLayer, {}),
	(DenseLayer, {'num_units': 64}),
	(DenseLayer, {'num_units': 10, 'nonlinearity': softmax}),
]

net1 = NeuralNet(
	layers=layers1,
	max_epochs=10,
	update_learning_rate=0.01,
	verbose=2,
)

# Show more information
net1.initialize()
layer_info = PrintLayerInfo()
layer_info(net1)

# train the net 
net1.fit(X, y)

##### Visualizations #####
from nolearn.lasagne.visualize import draw_to_notebook
from nolearn.lasagne.visualize import plot_loss
from nolearn.lasagne.visualize import plot_conv_weights
from nolearn.lasagne.visualize import plot_conv_activity
from nolearn.lasagne.visualize import plot_occlusion
from nolearn.lasagne.visualize import plot_saliency

draw_to_notebook(net1)
plt.savefig('results/net1_notbook.png')	# Must be front of `plt.show()`
plt.show()
plot_loss(net1)
plt.savefig('results/net1_plot_loss.png')	# Must be front of `plt.show()`
plt.show()
plot_conv_weights(net1.layers_[1], figsize=(4, 4))
plt.savefig('results/net1_plot_conv_weights.png')	# Must be front of `plt.show()`
plt.show()
x = X[0:1]
plot_conv_activity(net1.layers_[1], x)
plt.savefig('results/net1_plot_conv_activity.png')	# Must be front of `plt.show()`
plt.show()
plot_occlusion(net1, X[:5], y[:5])
plt.savefig('results/net1_plot_occlusion.png')	# Must be front of `plt.show()`
plt.show()
plot_saliency(net1, X[:5])
plt.savefig('results/net1_plot_saliency.png')	# Must be front of `plt.show()`
plt.show()

## test
#print "Start to test....."
#y_pred = net1.predict(X_test)
#print "The accuracy of this network is: %0.2f" % (y_pred == y_test).mean()
#
## store the network module
#import cPickle as pickle
#with open('results/net1.pickle','wb') as f:
#	pickle.dump(net1, f, -1)
