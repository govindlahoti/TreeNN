import csv 
import random
import threading
import numpy as np

from io import StringIO
from copy import deepcopy

from app.application import Application

class MNIST(Application):

	def __init__(self, **kwargs):
		# Method to initialize a Neural Network Object
		# Parameters
		# input_size - Size of the input layer
		# output_size - Size of the output layer
		# num_hidden_layers - Number of hidden layers in the neural network
		# hidden_layer_sizes - List of the hidden layer node sizes
		# alpha - learning rate
		# batchSize - Mini batch size
		# epochs - Number of epochs for training
		self.input_size = kwargs['input_size']
		self.output_size = kwargs['output_size']
		self.num_layers = kwargs['num_hidden_layers'] + 2
		self.layer_sizes = [self.input_size] + kwargs['hidden_layer_sizes'] + [self.output_size]

		self.alpha = kwargs['alpha']
		self.batch_size = kwargs['batch_size']
		self.epochs = kwargs['epochs']

		np.random.seed(42)
		random.seed(42)

		# Initializes the Neural Network Weights and Biases using a Normal Distribution with Mean 0 and Standard Deviation 1
		# weights - a list of matrices correspoding to the weights in various layers of the network
		# biases - corresponding list of biases for each layer
		self.weights = []
		self.biases = []

		for i in range(self.num_layers-1):
			size = self.layer_sizes[i], self.layer_sizes[i+1]
			self.biases.append(np.random.normal(0, 1, self.layer_sizes[i+1]))
			self.weights.append(np.random.normal(0,1,size))

		self.weights = np.asarray(self.weights)
		self.biases = np.asarray(self.biases)

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
		self.acquired_weights = []
		self.acquired_biases = []
		for i in range(self.num_layers-1):
			size = self.layer_sizes[i], self.layer_sizes[i+1]
			self.acquired_biases.append(np.zeros(self.layer_sizes[i+1]))
			self.acquired_weights.append(np.zeros(size))

		self.acquired_biases = np.asarray(self.acquired_biases)
		self.acquired_weights = np.asarray(self.acquired_weights)

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

	def train(self, training_data):
		# Method for training the Neural Network
		# Input
		# training_data - A list of training input data to the neural network
		# The methods trains the weights and baises using the training data(trainX, trainY)
		# and evaluates the validation set accuracy after each epoch of training

		for epoch in range(self.epochs):
			# Shuffle the training data for the current epoch
			training_data = np.asarray(training_data)
			rows, cols = training_data.shape
			X = training_data[:, range(cols-1)]>0
			Y = np.array(one_hot_encode_y(training_data[:, cols-1], 10))

			perm = np.arange(rows)
			np.random.shuffle(perm)
			X = X[perm]
			Y = Y[perm]

			# Initializing training loss and accuracy
			train_loss = 0
			train_acc = 0

			# Divide the training data into mini-batches
			num_batches = int(np.ceil(float(X.shape[0]) / self.batch_size))
			for batch_num in range(num_batches):

				X_batch = np.asarray(X[batch_num*self.batch_size: (batch_num+1)*self.batch_size])
				Y_batch = np.asarray(Y[batch_num*self.batch_size: (batch_num+1)*self.batch_size])

				# Calculate the activations after the feedforward pass
				activations = self.feedforward(X_batch)	

				# Compute the loss	
				loss = self.compute_loss(Y_batch, activations)
				train_loss += loss
				
				# Estimate the one-hot encoded predicted labels after the feedword pass
				pred_labels = one_hot_encode_y(np.argmax(activations[-1], axis=1), self.output_size)

				# Calculate the training accuracy for the current batch
				acc = self.compute_accuracy(Y_batch, pred_labels)
				train_acc += acc

				# Backpropagation Pass to adjust weights and biases of the neural network
				self.backpropagate(activations, Y_batch)

			# Print Training loss and accuracy statistics
			train_acc /= num_batches
			print("Epoch ", epoch, " Training Loss=", loss, " Training Accuracy=", train_acc)
			
	def compute_loss(self, Y, activations):
		# Returns the squared loss function given the activations and the true labels Y
		loss = (Y - activations[-1]) ** 2
		loss = np.mean(loss)
		return loss

	def compute_accuracy(self, Y, pred_labels):
		# Returns the accuracy given the true labels Y and predicted labels pred_labels
		correct = 0
		for i in range(len(Y)):
			if np.array_equal(Y[i], pred_labels[i]):
				correct += 1
		accuracy = (float(correct) / len(Y)) * 100
		return accuracy

	def evaluate(self, test_data):
		# Input 
		# test_data : Validation Input Data
		# Returns the validation accuracy evaluated over the current neural network model
		
		test_data = np.asarray(test_data)
		rows, cols = test_data.shape
		X = test_data[:, range(cols-1)]
		Y = np.array(one_hot_encode_y(test_data[:, cols-1], 10))

		activations = self.feedforward(X)
		pred = np.argmax(activations[-1], axis=1)
		valid_pred = one_hot_encode_y(pred, self.output_size)
		valid_acc = self.compute_accuracy(Y, valid_pred)
		return valid_acc

	def feedforward(self, X):
		# Input
		# X : Current Batch of Input Data as an nparray
		# Output
		# Returns the activations at each layer(starting from the first layer(input layer)) to 
		# the output layer of the network as a list of np arrays
		# Note: Activations at the first layer(input layer) is X itself
		
		activations = []

		activations.append(X)
		for i in range(1,self.num_layers):
			new_activation = np.matmul(activations[-1],self.weights[i-1]) + self.biases[i-1]
			new_activation = sigmoid(new_activation)
			activations.append(new_activation)

		return activations

	def backpropagate(self, activations, Y):
		# Input
		# activations : The activations at each layer(starting from second layer(first hidden layer)) of the
		# neural network calulated in the feedforward pass
		# Y : True labels of the training data
		# This method adjusts the weights(self.weights) and biases(self.biases) as calculated from the
		# backpropagation algorithm
		
		for i in range(len(activations),0,-1):
			if i == len(activations):
				target_delta = Y - activations[i-1]
				current_del = activations[i-1] * (1 - activations[i-1]) * target_delta
			else:
				self.parent_update_lock.acquire()
				self.biases[i-1] += self.alpha * np.sum(current_del,axis=0)
				self.acquired_biases[i-1] -= self.alpha * np.sum(current_del,axis=0)

				delta_w = np.matmul(np.transpose(activations[i-1]),current_del)
				target_delta = np.matmul(current_del,np.transpose(self.weights[i-1]))
				current_del = activations[i-1] * (1 - activations[i-1]) * target_delta
		
				self.weights[i-1] += self.alpha * delta_w 
				self.acquired_weights[i-1] -= self.alpha * delta_w
				self.parent_update_lock.release()

	@staticmethod
	def transform_sensor_data(data):
		return [list(map(int,rec)) for rec in csv.reader(StringIO(data), delimiter=',')]

	@staticmethod
	def get_test_data(test_file_handler, filesize, skipdata, size=200):
		"""
		Get test data from file. 
		Using random-seeking in file to limit RAM usage
		"""		
		test_file_handler.seek(0)
		return [list(map(int,rec)) for rec in csv.reader(test_file_handler, delimiter=',')]

def one_hot_encode_y(Y, nb_classes):
	# Calculates one-hot encoding for a given list of labels
	# Input :- Y : An integer or a list of labels
	# Output :- Coreesponding one hot encoded vector or the list of one-hot encoded vectors
	return (np.eye(nb_classes)[Y]).astype(int)

def sigmoid(x):
	# Calculates the sigmoid function
	X = np.copy(x)
	return 1 / (1 + np.exp(-X))
		
