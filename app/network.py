# Citation: Part of the code is taken from publicly avaiable implementation on
# http://neuralnetworksanddeeplearning.com/chap1.html
#check the dataprep file and modify the code. Right now the file may contain null values for the spatial neigbhborhood and flag is associated at each row to know whether it labelled
# or unlabelled data ..................we need to choose the function based on the input row.
# a batch may contain labelled and unlabelled both type of rows  
import random
import numpy as np
import threading
import timeit
import time
import sys
import csv
from copy import deepcopy

from app.application import Application

# The class of DNN
def dist(x,y):  
	z = np.sqrt(np.sum((x-y)**2))
	return np.exp(-z)

def mapToFloat(x):
	res = [] 
	for k in x.replace('(','').replace(')','').split(','):
		k = k.replace('\'','')
		try: 
			res.append(float(k))
		except:
			res.append(0.0)

	res = np.array(res)
	return res.reshape((len(res),1))

class Network(Application):

	def __init__(self, sizes):
		"""
		The list 'sizes' contains the number of neurons in the
		respective layers of the network.  For example, if the list
		was [2, 3, 1] then it would be a three-layer network, with the
		first layer containing 2 neurons, the second layer 3 neurons,
		and the third layer 1 neuron.  The biases and weights for the
		network are initialized randomly, using a Gaussian
		distribution with mean 0, and variance 1.  Note that the first
		layer is assumed to be an input layer, and by convention we
		won't set any biases for those neurons, since biases are only
		ever used in computing the outputs from later layers."""
		self.num_layers = len(sizes)
		self.sizes = sizes
		
		np.random.seed(42)
		random.seed(42)

		self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
		self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

		self._reset_acquired_weights_and_biases()
		self.parent_update_lock = threading.Lock()

	def get_model(self):
		"""Return the present model (weights and biases)"""
		self.parent_update_lock.acquire()
		weights = deepcopy(self.weights)
		biases = deepcopy(self.biases)
		self.parent_update_lock.release()
		return [weights, biases]

	def _reset_acquired_weights_and_biases(self):
		"""Reset the acquired weights to zeros"""
		self.acquired_biases = [np.zeros((y, 1)) for y in self.sizes[1:]]
		self.acquired_weights = [np.zeros((y, x)) for x, y in zip(self.sizes[:-1], self.sizes[1:])]

	def apply_kid_gradient(self, weight_gradient, bias_gradient):
		"""Update the model (weights, biases) by adding the gradients obtained from the child node"""
		self.parent_update_lock.acquire()

		self.weights = [w - wg for w, wg in zip(self.weights, weight_gradient)]
		self.biases = [b - bg for b, bg in zip(self.biases, bias_gradient)]
		self.acquired_weights = [w + wg for w, wg in zip(self.acquired_weights, weight_gradient)]
		self.acquired_biases = [b + bg for b, bg in zip(self.acquired_biases, bias_gradient)]

		self.parent_update_lock.release()


	def use_parent_model(self, weight, bias):
		"""Replace own model completely by the parent model"""
		self.parent_update_lock.acquire()
		self.weights = weight
		self.biases = bias
		self._reset_acquired_weights_and_biases()
		self.parent_update_lock.release()


	def get_and_reset_acquired_gradients(self):
		"""Return the acquired gradients in the model. Also reset this to zero"""
		self.parent_update_lock.acquire()
		acquired_weights = deepcopy(self.acquired_weights)
		acquired_biases = deepcopy(self.acquired_biases)
		self._reset_acquired_weights_and_biases()
		self.parent_update_lock.release()
		return [acquired_weights, acquired_biases]

	def feedforward(self, a, return_activations=True):
		"""Return the output of the network if ``a`` is input."""
		activations = [a]
		activation = a
		zs = [] 
		for b, w in zip(self.biases, self.weights):
			z = np.dot(w, activation) 
			z.shape = (len(z),1)
			z = z + b
			zs.append(z)
			activation = sigmoid(z)
			activations.append(activation)

		if return_activations:
			return zs, activations
		else:
			return activation
		
	def train(self, training_data, **kwargs):
		"""
		Train the neural network using mini-batch stochastic
		gradient descent.  The ``training_data`` is a list of tuples
		``(x, y)`` representing the training inputs and the desired
		outputs.  The other non-optional parameters are
		self-explanatory.  If ``test_data`` is provided then the
		network will be evaluated against the test data after each
		epoch, and partial progress printed out.  This is useful for
		tracking progress, but slows things down substantially.
		"""

		epochs = kwargs['epochs'] 
		mini_batch_size = kwargs['mini_batch_size']
		eta = kwargs['eta']
		lmbda = kwargs['lmbda']
		alpha = kwargs['alpha']
		beta = kwargs['beta']

		self.beta = 0.9
		self.V_b = [np.zeros(b.shape) for b in self.biases]
		self.V_w = [np.zeros(w.shape) for w in self.weights]
		
		n = len(training_data)
		for j in range(epochs):
			random.shuffle(training_data)
			mini_batches = [
				training_data[k:k+mini_batch_size]
				for k in range(0, n, mini_batch_size)]
			i = 0

			for mini_batch in mini_batches:
				self.update_mini_batch(mini_batch, eta, lmbda, alpha, beta, n)

	def update_mini_batch(self, mini_batch, eta, lmbda, alpha, beta, n):
		"""Update the network's weights and biases by applying
		gradient descent using backpropagation to a single mini batch.
		The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
		is the learning rate. L2 regularization is added"""
		nabla_b = [np.zeros(b.shape) for b in self.biases] 
		nabla_w = [np.zeros(w.shape) for w in self.weights]

		nabla_bl = [np.zeros(b.shape) for b in self.biases]
		nabla_wl = [np.zeros(w.shape) for w in self.weights]

		data_count = len(mini_batch)
		labelled_count = 0

		for x, xs1, xs2, xt1, xt2, y, flag in mini_batch:
			x = mapToFloat(x)
			xs1 = mapToFloat(xs1)
			xs2 = mapToFloat(xs2)
			xt1 = mapToFloat(xt1)
			xt2 = mapToFloat(xt2)
			y = mapToFloat(y)
			size_bool = len(x) == 276 and len(xs1) == 276 and len(xs2) == 276 and len(xt1) == 276 and len(xt2) == 276
			flag = flag[1:-2]

			delta_nabla_b, delta_nabla_w = None, None
			## Labelled data
			if(flag == "1" and len(y)==48 and size_bool):
				delta_nabla_b, delta_nabla_w, delta_nabla_bl, delta_nabla_wl = self.backprop(x, xs1, xs2, xt1, xt2, alpha, beta, True, y)
				nabla_bl = [nbl+dnbl for nbl, dnbl in zip(nabla_bl, delta_nabla_bl)]
				nabla_wl = [nwl+dnwl for nwl, dnwl in zip(nabla_wl, delta_nabla_wl)]

			## Unlabelled data
			elif(flag == "0" and size_bool):
				delta_nabla_b, delta_nabla_w,_,_ = self.backprop(x, xs1, xs2, xt1, xt2, alpha, beta, False)
				
			else:
				print("Unknown Data format")
				exit(0)

			nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
			nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]        

		nab_b = [ nb/data_count for nb in nabla_b]
		nab_w = [ lmbda*w + nw/data_count for w, nw in zip(self.weights, nabla_w)]

		if(labelled_count>0): 
			nab_b = [ b + nbl/labelled_count for b, nbl in zip(nab_b, nabla_bl)]
			nab_w = [ w + nwl/labelled_count for w, nwl in zip(nab_w, nabla_wl)]

		self.parent_update_lock.acquire()
		
		self.V_b = [ self.beta*vb + (1-self.beta)*eta*nb for vb, nb in zip(self.V_b, nab_b)]
		self.V_w = [ self.beta*vw + (1-self.beta)*eta*nw for vw, nw in zip(self.V_w, nab_w)]

		self.biases =   [ b - vb for b, vb in zip(self.biases, self.V_b)]
		self.weights =  [ w - vw for w, vw in zip(self.weights, self.V_w)]

		self.acquired_biases =  [ b + vb for b, vb in zip(self.acquired_biases, self.V_b)]
		self.acquired_weights = [ w + vw for w, vw in zip(self.acquired_weights, self.V_w)]
				
		self.parent_update_lock.release()    
	
	def compute_neighbourhood_delta(self, a11, z_n, a21, z, a_n, a):
		delta1 = a11 * sigmoid_prime(z_n) 
		delta2 = a21 * sigmoid_prime(z)

		d1 = np.dot(delta1, a_n.transpose())
		d2 = np.dot(delta2, a.transpose())

		return delta1, delta2, d2-d1

	def backprop(self, x, xs1, xs2, xt1, xt2, alpha, beta, is_labelled, y=None):
		"""
		Return a tuple ``(nabla_b, nabla_w)`` representing the
		gradient for the cost function C_x.  ``nabla_b`` and
		``nabla_w`` are layer-by-layer lists of numpy arrays, similar
		to ``self.biases`` and ``self.weights``.
		"""
		
		nabla_b = [np.zeros(b.shape) for b in self.biases]
		nabla_w = [np.zeros(w.shape) for w in self.weights]
		nabla_bl = [np.zeros(b.shape) for b in self.biases]
		nabla_wl = [np.zeros(w.shape) for w in self.weights]
		
		zs,activations = self.feedforward(x)

		neighborhood = [xs1,xs2,xt1,xt2]       
		d = [dist(x,k) for k in neighborhood]        
		[zs_n,a] = zip(*[self.feedforward(k) for k in neighborhood])
		
		delta_n = [[0.0 for i in range(4)] for j in range(4)]
		diff_delta = [0 for i in range(4)]
		delta = 0

		for l in range(1, self.num_layers):
			if is_labelled:
				delta = self.derivative(l, output_activations=activations[-1], y=y, delta_n=delta) * sigmoid_prime(zs[-l])
				# print(delta.shape)
			
			for i in range(4):
				a11 = self.derivative(l, output_activations=activations[-1], y=a[i][-1], delta_n=delta_n[i][0])
				a21 = self.derivative(l, output_activations=activations[-1], y=a[i][-1], delta_n=delta_n[i][1])
				delta_n[i][0], delta_n[i][1],diff_delta[i] = self.compute_neighbourhood_delta(a11, zs_n[i][-l], a21, zs[-l], a[i][-l-1], activations[-l-1])
			
			nabla_b[-l] = alpha*(d[0]*(delta_n[0][1] - delta_n[0][0]) + d[1]*(delta_n[1][1] - delta_n[1][0])) + beta*(d[2]*(delta_n[2][1] - delta_n[2][0]) + d[3]*(delta_n[3][1] - delta_n[3][0]))
			nabla_w[-l] = alpha*(d[0]*diff_delta[0] + d[1]*diff_delta[1]) + beta*(d[2]*diff_delta[2] + d[3]*diff_delta[3])
			
			nabla_bl[-l] = delta 
			nabla_wl[-l] = np.dot(delta, activations[-l-1].transpose())

		return (nabla_b, nabla_w, nabla_bl, nabla_wl)
	
	def evaluate(self, test_data):
		"""
		Return the number of test inputs for which the neural
		network outputs the correct result. Note that the neural
		network's output is assumed to be the index of whichever
		neuron in the final layer has the highest activation.
		"""
		if(len(test_data[0]) == 7):            
			test_results = [(self.feedforward(mapToFloat(x),return_activations=False), mapToFloat(y))
						for (x, xs1, xs2, xt1, xt2, y,flag) in test_data]
		else:
			if(len(test_data[0])==2):
				test_results = [(self.feedforward(mapToFloat(x),return_activations=False), mapToFloat(y))
						for (x, y) in test_data]
			else:
				test_results = [(self.feedforward(mapToFloat(x),return_activations=False), 0.0)
						for (x, xs1,xs2,xt1,xt2) in test_data]    

		r,n = 0,0
		for x,y in test_results:
			if len(y) == 48:
				r += np.linalg.norm(x-y)**2
				n += 1
		return np.sqrt(1.*r/n)
		# return sum([ if len(y)==48 else 0 for x,y in test_results])

	def derivative(self, l, output_activations=None, y=None, delta_n=None):
		"""
		Return the vector of partial derivatives \partial C_x /
		\partial a for the output activations.
		"""
		if l==1:
			return (output_activations-y)
		else:
			return np.dot(self.weights[-l+1].transpose(), delta_n)


def sigmoid(z):
	"""The sigmoid function."""
	return 1.0/(1.0+np.exp(-z))


def sigmoid_prime(z):
	"""Derivative of the sigmoid function."""
	return sigmoid(z)*(1-sigmoid(z))
