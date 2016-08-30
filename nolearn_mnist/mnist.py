#Source: http://nbviewer.jupyter.org/github/dnouri/nolearn/blob/master/docs/notebooks/CNN_tutorial.ipynb
import os
import matplotlib.pyplot as plt
import numpy as np

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

def load_mnist(path):
	X = []
	y = []
	with open(path, 'rb') as f:
		next(f)  # skip header
		for line in f:
			yi, xi = line.split(',', 1)
			y.append(yi)
			X.append(xi.split(','))

	# Theano works with f[32 precision
	X = np.array(X).astype(np.float32)
	y = np.array(y).astype(np.int32)
	# apply some very simple normalization to the data

	X -= X.mean()
	X /= X.std()
	
	# For convolutional layers, the default shape of data is bc01,
	# i.e. batch size x color channels x image dimension 1 x image dimension 2.
	# Therefore, we reshape the X data to -1, 1, 28, 28.
	X = X.reshape(
		-1,  # number of samples, -1 makes it so that this number is determined automatically
		1,   # 1 color channel, since images are only black and white
		28,  # first image dimension (vertical)
		28,  # second image dimension (horizontal)
	)
	return X, y

# here you should enter the path to your MNIST data
path_train = os.path.join(os.path.expanduser('~'), 'data/mnist/train.csv')
path_test = os.path.join(os.path.expanduser('~'), 'data/mnist/test.csv')

X, y = load_mnist(path_train)
X_test, y_test = load_mnist(path_test)

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

layers0 = [
	# layer dealing with the input data
	(InputLayer, {'shape': (None, X.shape[1], X.shape[2], X.shape[3])}),

	# first stage of our convolutional layers
	(Conv2DLayer, {'num_filters': 96, 'filter_size': 5}),
	(Conv2DLayer, {'num_filters': 96, 'filter_size': 3}),
	(Conv2DLayer, {'num_filters': 96, 'filter_size': 3}),
	(Conv2DLayer, {'num_filters': 96, 'filter_size': 3}),
	(Conv2DLayer, {'num_filters': 96, 'filter_size': 3}),
	(MaxPool2DLayer, {'pool_size': 2}),
	
	# second stage of our convolutional layers
	(Conv2DLayer, {'num_filters': 128, 'filter_size': 3}),
	(Conv2DLayer, {'num_filters': 128, 'filter_size': 3}),
	(Conv2DLayer, {'num_filters': 128, 'filter_size': 3}),
	(MaxPool2DLayer, {'pool_size': 2}),
	
	# two dense layers with dropout
	(DenseLayer, {'num_units': 64}),
	(DropoutLayer, {}),
	(DenseLayer, {'num_units': 64}),
	
	# the output layer
	(DenseLayer, {'num_units': 10, 'nonlinearity': softmax}),
]

net0 = NeuralNet(
    layers=layers0,
    max_epochs=10,

    update=adam,
    update_learning_rate=0.0002,

    objective_l2=0.0025,

    train_split=TrainSplit(eval_size=0.25),
    verbose=1,
)
# Show more information
net0.initialize()
layer_info = PrintLayerInfo()
layer_info(net0)

# train the net 
net0.fit(X, y)

# test
print "Start to test....."
y_pred = net0.predict(X_test)
print "The accuracy of this network is: %0.2f" % (y.pred == y_test).mean()

##### Visualizations #####
from nolearn.lasagne.visualize import draw_to_notebook
from nolearn.lasagne.visualize import plot_loss
from nolearn.lasagne.visualize import plot_conv_weights
from nolearn.lasagne.visualize import plot_conv_activity
from nolearn.lasagne.visualize import plot_occlusion
from nolearn.lasagne.visualize import plot_saliency

draw_to_notebook(net0)
plot_loss(net0)
plot_conv_weights(net0.layers_[1], figsize=(4, 4))
x = X[0:1]
plot_conv_activity(net0.layers_[1], x)
plot_occlusion(net0, X[:5], y[:5])
plot_saliency(net0, X[:5])

# store the network module
import cPickle as pickle
with open('results/net0.pickle','wb') as f:
	pickle.dump(net0, f, -1)

