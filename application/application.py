"""
An abstract class for instantiating any application which can be simulated by the Simulator
"""

from abc import ABC, abstractmethod

class Application(ABC):

	#------------------ For Parameter Servers -------------------
	@abstractmethod
	def get_model(self):
		"""
		Return the present model. Used by Parameter servers for merging gradients
		For supervised learning, the weights and biases are returned as 
			they are sufficient in representing the model. ie. Input*w + b = Output
		"""
		print("Method not implemented!")

	@abstractmethod
	def apply_kid_gradient(self, weight_gradient, bias_gradient):
		"""
		Update the model (weights, biases) by adding the gradients obtained from the child node
		"""
		print("Method not implemented!")

	#------------------ For Worker nodes ------------------------
	@abstractmethod
	def train(self,training_data, epochs, batch_size):
		"""
		Method which defines the training algorithm used by the application.
		In case of AQI, the algorithm is Stochastic Gradient Descent 
		"""
		print("Method not implemented!")

	#------------------ For Both --------------------------------
	@abstractmethod
	def use_parent_model(self, weight, bias):
		"""
		Fetch model from parent node
		"""
		print("Method not implemented!")

	@abstractmethod
	def get_and_reset_acquired_gradients(self):
		"""
		Return the acquired gradients in the model. Also reset these to zero
		"""
		print("Method not implemented!")

	@abstractmethod
	def evaluate(self,test_data):
		"""
		Evaluate the current model using test_data
		"""
		print("Method not implemented!")

