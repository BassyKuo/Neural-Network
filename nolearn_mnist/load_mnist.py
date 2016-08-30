#Source: http://nbviewer.jupyter.org/github/dnouri/nolearn/blob/master/docs/notebooks/CNN_tutorial.ipynb
import os
import sys
import gzip
import numpy as np
if sys.version_info[0] == 2:
	from urllib import urlretrieve
else:
	from urllib.request import urlretrieve

#X_train = get_data('train-images-idx3-ubyte.gz')
#y_train = get_data('train-labels-idx1-ubyte.gz')
#X_test  = get_data('t10k-images-idx3-ubyte.gz')
#y_test  = get_data('t10k-labels-idx1-ubyte.gz')


def get_data(filename, source='http://yann.lecun.com/exdb/mnist/', np_offset=16):
	folder = "data/"
	filepath = os.path.join(folder + filename)
	if not os.path.exists(folder):
		os.mkdir(folder)
	if not os.path.exists(filepath):
		print("Downloading %s" % filename)
		urlretrieve(source + filename, filepath)
	with gzip.open(filepath, 'rb') as f:
	    data = np.frombuffer(f.read(), np.uint8, offset=np_offset)
	return data

def load_mnist(path_x, path_y):
	X, y = [], []
	X = get_data(path_x,np_offset=16).reshape(-1,1,28,28).astype(np.float32) 
	y = get_data(path_y,np_offset=8).astype(np.int32)

	X = ( X - X.mean() ) / X.std()
	
	# For convolutional layers, the default shape of data is bc01,
	# i.e. batch size x color channels x image dimension 1 x image dimension 2.
	# Therefore, we reshape the X data to -1, 1, 28, 28.
	#X.reshape(
	#	-1,  # number of samples, -1 makes it so that this number is determined automatically
	#	1,   # 1 color channel, since images are only black and white
	#	28,  # first image dimension (vertical)
	#	28,  # second image dimension (horizontal)
	#)
	return X, y

def load_mnist_set():
	path_x_train = 'train-images-idx3-ubyte.gz'
	path_y_train = 'train-labels-idx1-ubyte.gz'
	path_x_test  = 't10k-images-idx3-ubyte.gz'
	path_y_test	 = 't10k-labels-idx1-ubyte.gz'
	
	X_train, y_train = load_mnist(path_x_train, path_y_train)
	X_test, y_test = load_mnist(path_x_test, path_y_test)

	return X_train, y_train, X_test, y_test
	
if __name__ == '__main__':
	X_train, y_train, X_test, y_test = load_mnist_set()
	
	print "X_train.shape: ", X_train.shape
	print "y_train.shape: ", y_train.shape
	print "X_test.shape: ", X_test.shape
	print "y_test.shape: ", y_test.shape
