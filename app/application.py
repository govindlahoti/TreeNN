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
	def train(self,training_data):
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

	#----------------- For Sensor Data Processing ------------
	@abstractmethod
	def transform_sensor_data(data):
		"""
		Transforms sensor data so that it can be trained for learning the model
		"""
		print("Method not implemented!")

	@abstractmethod
	def get_test_data(test_file_handler, filesize, skipdata, size):
		"""
		Get test data from file. 
		Using random-seeking in file to limit RAM usage
		"""	
		print("Method not implemented!")

