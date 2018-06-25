from SimpleXMLRPCServer import SimpleXMLRPCServer
import time
from Queue import Queue
import xmlrpclib
import sys
import thread
import csv
from distlib.util import CSVReader
from network import *


# The logic for generating data
#np.random.seed(42)
#data_trend = np.random.normal(0, 1, (48, 729))

#def get_data(random=False):
#	"""Returns the randomly generated data"""
#	def get_y(x):
#		return sigmoid(np.dot(data_trend, x))
#	
#	if not random:
#		np.random.seed(42)
#	
#	x_vals = [np.random.normal(0, 1, (729, 1)) for _ in range(20)]
#	y_vals = map(get_y, x_vals)
#
#	return zip(x_vals, y_vals)

# read data of sensors from file
def get_data(fileName):
	with open(fileName) as csv_file:
		print "reading data"
		csv_reader = csv.reader(csv_file)
		train_data = list(csv_reader)
		print "train data", len(train_data)
	
	return train_data




def get_size(l):
	"""Returns the size of list/tuple (even if it is nested)"""
	if type(l) == list or type(l) == tuple:
		return sum(get_size(subl) for subl in l)
	else:
		return 1

# The class for simulating an edge device
class Node:

	def __init__(self, data):
		"""Constructor. Fillin all the required information from the 
		metadata passed to it. 
		Spawns three threads - 
		1. To consume the gradients being obtained from the child node
		2. To run the RPC server
		3. To run the downpour SGD algoritm (sharing logic)
		"""
		self.id = data['id']
		self.own_address = data['own_address']
		self.parent_id = data['parent_id']
		self.parent_address = data['parent_address']
		self.is_worker = data['is_worker']
		self.connected_with_parent = False
		self.e = 0  # how many epoches 
		self.prev_e = 0 # 
		self.e_lock = threading.Lock()
		self.push_interval = data['push_interval']
		self.pull_interval = data['pull_interval']
		self.delays = data['delays']
		self.inputfile = data['file_name']
		print self.inputfile
		
		self.network = Network([276, 276, 276, 1])

		self.log_file = open(str(self.id) + '.log', 'a')
		
		self.acquired_gradients_from_kids = Queue()                          # what does it mean ????

		thread.start_new_thread(self.comsume_gradients_from_kids_thread, ())
		thread.start_new_thread(self.run_rpc_server_thread, ())
		thread.start_new_thread(self.run_sharing_logic_thread, ())


	def get_parent(self):
		"""
		Connect with the parent's RPC server if not already connected. 
		And then return the connection object
		"""
		time.sleep(self.delays[self.parent_id] / 1000.)
		if self.connected_with_parent:
			return self.parent

		elif not self.parent_address:
			return None
		
		else:
			self.parent = xmlrpclib.ServerProxy(self.parent_address, allow_none=True)
			
			while True:
				self.log('Node {} : Waiting for parent {} ...'.format(self.id, self.parent_address))
				try:
					self.parent.pull_from_child(self.id)
					self.log('Node {} : Connected with parent {}'.format(self.id, self.parent_address))
					self.connected_with_parent = True
					break
				except Exception, e:
					print e
					time.sleep(1)

			return self.parent


	def push_from_child(self, weight_gradient, bias_gradient, child_id):
		"""RPC function. Add the graients obtained from child node into the queue"""
		self.log('Got gradients from child ' + str(child_id))
		self.log('Network cost incurrec = ' + str(get_size([weight_gradient, bias_gradient])))
		weight_gradient = [np.array(x) for x in weight_gradient]
		bias_gradient = [np.array(x) for x in bias_gradient]
		self.acquired_gradients_from_kids.put([weight_gradient, bias_gradient])
	

	def pull_from_child(self, child_id):
		"""RPC function. Return the model (weights and biases) to the child node"""
		self.log('Got pull request from child ' + str(child_id))
		model = self.network.get_model()
		model[0] = [x.tolist() for x in model[0]]
		model[1] = [x.tolist() for x in model[1]]
		self.log('Network cost incurred = ' + str(get_size(model)))
		return model
	

	def push_to_parent(self, weight_gradient, bias_gradient):
		"""Push the gradients to the parent node. Calls parent's RPC function internally"""
		if not self.parent_address:
			return
		self.log('Sending gradients to parent' + str(self.parent_id))
		weight_gradient = [x.tolist() for x in weight_gradient]
		bias_gradient = [x.tolist() for x in bias_gradient]
		self.get_parent().push_from_child(weight_gradient, bias_gradient, self.id)


	def pull_from_parent(self):
		"""Pull the mode from the parent node. Calls parent's RPC function internally"""
		if not self.parent_address:
			return
		model = self.get_parent().pull_from_child(self.id)
		model[0] = [np.array(x) for x in model[0]]
		model[1] = [np.array(x) for x in model[1]]
		self.log('Got model from parent ' + str(self.parent_id))
		return model

	def get_loss(self):
		"""RPC function. Return the loss of the present model at the node"""
		return float(self.network.evaluate(data))

	def comsume_gradients_from_kids_thread(self):
		"""This thread runs indefinitely. It montors the queue of acquired gradients from the 
		child nodes. It modifies the node's model using those gradients.
		"""
		while True:
			weight_gradient, bias_gradient = self.acquired_gradients_from_kids.get()
			self.network.apply_kid_gradient(weight_gradient, bias_gradient)
			self.e_lock.acquire()
			self.e += 1
			self.e_lock.release()


	def run_rpc_server_thread(self):
		"""Thread to run the RPC server for the node"""
		server = SimpleXMLRPCServer(self.own_address, allow_none=True)
		server.register_function(self.push_from_child, "push_from_child")
		server.register_function(self.pull_from_child, "pull_from_child")
		server.register_function(self.get_loss, "get_loss")
		# add functions for communication between workers
		server.serve_forever()


	def run_sharing_logic_thread(self):
		"""Runs the downpour SGD logic for sharing the weights/models with the parent and child
		nodes. Leaf nodes train the model using the data while the non-leaf nodes don't. Refer the
		BTP Report to understand the downpour SGD logic
		"""
		while True:
			data = get_data(self.inputfile)
			
			if self.is_worker:

				if self.e % self.pull_interval == 0:
					if self.parent_address:
						self.network.use_parent_model(*self.pull_from_parent())

				self.network.SGD(data, epochs=1)

				if self.e % self.push_interval == 0:
					if self.parent_address:
						self.push_to_parent(*self.network.get_and_reset_acquired_gradients())

				self.e += 1
		
			else:
				self.e_lock.acquire()
				e = self.e
				self.e_lock.release()
				
				if e != self.prev_e:
					self.prev_e = e
				else:
					time.sleep(1)
					continue
				
				if e % self.pull_interval == 0:
					if self.parent_address:
						self.network.use_parent_model(*self.pull_from_parent())                     # this code is run by non-worker node

				if e % self.push_interval == 0:
					if self.parent_address:
						self.push_to_parent(*self.network.get_and_reset_acquired_gradients())
              
			self.log('Epoch {} Loss = {}'.format(self.e, self.network.evaluate(data)))


	def log(self, s):
		"""Write s to the log file of this node"""
		self.log_file.write(s +'\n')
		self.log_file.flush()
