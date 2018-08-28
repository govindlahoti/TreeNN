import sys
import time
import threading

import csv
import json
from collections import OrderedDict
from distlib.util import CSVReader

from network import *

from abc import ABC, abstractmethod

from xmlrpc.client import ServerProxy
from xmlrpc.server import SimpleXMLRPCServer

# read data of sensors from file
def get_data(fileName):
	with open(fileName) as csv_file:
		print("reading data")
		csv_reader = csv.reader(csv_file)
		train_data = list(csv_reader)
		print("train data", len(train_data))
	
	return train_data

def get_size(l):
	"""Returns the size of list/tuple (even if it is nested)"""
	if type(l) == list or type(l) == tuple:
		return sum(get_size(subl) for subl in l)
	else:
		return 8	### Float size = 8 bytes

# An abstract class for simulating an edge device
class Node(ABC):

	def __init__(self, data):
		"""
		Constructor. Fill in all the required information from the 
		metadata passed to it. 
		Spawns three threads - 
		1. To consume the gradients being obtained from the child node
		2. To run the RPC server
		3. To run the downpour SGD algoritm (sharing logic)
		"""

		### Information about self and parent
		self.id = data['id']
		self.own_tuple_address = tuple(data['own_address'])
		self.own_server_address = 'http://%s:%d'%self.own_tuple_address
		self.parent_id = data['parent_id']
		self.parent_address = data['parent_address']
		
		### Information about connections 
		self.connection_objs = {}
		self.addresses = { int(k):v for k,v in data['addresses'].items() }
		for w in data['delays']:
			self.connection_objs[int(w)] = None
		self.push_interval = data['push_interval']
		self.pull_interval = data['pull_interval']
		self.delays = { int(k):v for k,v in data['delays'].items() }

		### Information about the learning model
		self.epoch_limit = data['epoch_limit']
		self.inputfile = data['file_name']
		self.batch_size = data['batch_size']
		print(self.inputfile)
		self.network = Network([276, 276, 276, 48])

		### Meta
		self.log_file = open(str(self.id) + '.log', 'a')
		self.master_address = data['master_address']
		self.master = None
	
	###------------------------- Abstract Methods ---------------------------------------------
	@abstractmethod
	def init_threads(self):
		print("Method not implemented")

	### Server thread 
	@abstractmethod
	def run_rpc_server_thread(self):
		"""Thread to run the RPC server for the node"""
		print("Method not implemented")		

	###------------------------- Connection functions -----------------------------------------
	def get_master(self):
		if self.master is not None :
			return self.master

		self.master = ServerProxy(self.master_address, allow_none=True)
		return self.master

	def get_parent(self):
		"""
		Connect with the parent's RPC server if not already connected. 
		And then return the connection object
		"""
		# print(self.parent_id,self.addresses)
		return self.get_node(self.parent_id)

	def get_node(self, node_id):
		"""
		Connect with the node's RPC server if not already connected. 
		And then return the connection object
		"""
		time.sleep(self.delays[node_id] / 1000.)
		
		if self.connection_objs[node_id] is not None:
			return self.connection_objs[node_id]

		elif not self.addresses[node_id]:
			return None

		else:
			self.connection_objs[node_id] = ServerProxy(self.addresses[node_id], allow_none=True)
			print(self.addresses[node_id],self.connection_objs[node_id])
			while True:
				self.log(self.create_log('CONN','Waiting for node %s to connect'%(node_id)))
				try:
					self.connection_objs[node_id].recv_message(self.id, 'hello')
					self.log(self.create_log('CONN','Connected with node %s'%(node_id)))
					break
				except Exception as e:
					print(e)
					time.sleep(1)

			return self.connection_objs[node_id]		
	
	### Communication with parent
	def push_to_parent(self, weight_gradient, bias_gradient):
		"""Push the gradients to the parent node. Calls parent's RPC function internally"""
		if not self.parent_address:
			return

		self.log(self.create_log('CONN','Sending gradients to parent %d'%(self.parent_id)))
		weight_gradient = [x.tolist() for x in weight_gradient]
		bias_gradient = [x.tolist() for x in bias_gradient]

		self.log(self.create_log('CONN',{
									'Network cost incurred in bytes' : get_size([weight_gradient, bias_gradient])
								}))

		self.get_parent().push_from_child(weight_gradient, bias_gradient, self.id)

	def pull_from_parent(self):
		"""Pull the mode from the parent node. Calls parent's RPC function internally"""
		if not self.parent_address:
			return
		model = self.get_parent().pull_from_child(self.id)
		model[0] = [np.array(x) for x in model[0]]
		model[1] = [np.array(x) for x in model[1]]
		
		self.log(self.create_log('CONN','Got model from parent %d'%(self.parent_id)))
		self.log(self.create_log('CONN',{
									'Network cost incurred in bytes' : get_size(model)
								}))
		
		return model

	###-------------------------- RPC functions -----------------------------------------
	def get_loss(self):
		"""RPC function. Return the loss of the present model at the node"""
		return float(self.network.evaluate(data))

	def recv_message(self, sender_id, msg):
		"""Add the logic of what is to be done upon	reciept of message from some other worker"""
		self.log(self.create_log('CONN','Received message from node id %d, msg: %s'%(sender_id, msg)))		
	
	def remote_shutdown(self):
		t = threading.Thread(target=self.shutdown_thread)
		t.start();t.join()

	def shutdown_thread(self):
		self.server.shutdown()

	### Meta functions
	def send_message(self, receiver_id, msg):
		"""Use this function to send the data to whichever node you wish to"""
		self.log('Sending message to node_id {}, msg: {}'.format(receiver_id, msg))
		self.get_node(receiver_id).recv_message(self.id, msg)
	
	def create_log(self, log_type, payload):
		log = OrderedDict({
				'node_id'	: self.id,
				'type'		: log_type,
				'payload'	: payload
			})
		return json.dumps(log)

	def log(self, s):
		"""Write s to the log file of this node"""
		self.get_master().log_report(s)
		self.log_file.write(s +'\n')
		self.log_file.flush()



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
