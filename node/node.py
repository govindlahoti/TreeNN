"""
An abstract class for simulating an edge device
"""

import sys
import time
import threading

import os
import csv
import json
import random
import numpy as np
from collections import OrderedDict

from app import *
from policy import *
from utility.const import *

from abc import ABC, abstractmethod

from xmlrpc.client import ServerProxy
from xmlrpc.server import SimpleXMLRPCServer

def get_size(l):
	"""
	Returns the size of list/tuple (even if it is nested)
	Used by node.py
	"""

	if type(l) == list or type(l) == tuple:
		return sum(get_size(subl) for subl in l)
	else:
		return 8	### Float size = 8 bytes


class Node(ABC):

	def __init__(self, data):
		"""
		Constructor. Fill in all the required information from the 
		metadata passed to it. 
		"""

		### Information about self and parent
		self.id = data['id']
		self.own_tuple_address = tuple(data['own_address'])
		self.own_server_address = 'http://%s:%d'%self.own_tuple_address
		self.policy = eval(data['policy']['type'])(**data['policy']['args'])
		self.parent_id = data['parent_id']
		self.parent_address = data['parent_address']
		
		### Information about connections 
		self.connection_objs = {}
		self.addresses = { int(k):v for k,v in data['addresses'].items() }
		for w in data['bandwidths']:
			self.connection_objs[int(w)] = None
		self.bandwidths = { int(k):v for k,v in data['bandwidths'].items() }

		### Information about the test data and learning model
		self.skiptestdata = 0
		self.test_files = os.listdir(data['test_directory'])
		self.test_file_handlers = { f:open(data['test_directory']+f, 'r') for f in self.test_files}

		def get_filesize(test_file):
			test_file.seek(0, os.SEEK_END)
			return test_file.tell()

		self.filesizes = { f:get_filesize(self.test_file_handlers[f]) for f in self.test_files}

		self.application = eval(data['application_arguments']['model'])(**data['application_arguments'])

		### Meta
		self.log_file = open('logs/%d.log'%self.id, 'a')
		self.master_address = data['master_address']
		self.master = None
	
	###------------------------- Abstract Methods ---------------------------------------------
	@abstractmethod
	def init_threads(self):
		print("Method not implemented")

	### Server thread 
	@abstractmethod
	def run_rpc_server_thread(self):
		"""
		Thread to run the RPC server for the node
		"""
		print("Method not implemented")	

	@abstractmethod
	def receive_message(self, sender_id, msg):
		"""
		Add the logic of what is to be done upon the receival of message from some other node
		"""
		print("Method not implemented")

	###------------------------- Connection functions -----------------------------------------
	def get_master(self):
		"""
		Connect with master's RPC server for sending logs
		"""

		if self.master is not None :
			return self.master

		self.master = ServerProxy(self.master_address, allow_none=True)
		return self.master

	def get_parent(self):
		"""
		Connect with the parent's RPC server if not already connected. 
		And then return the connection object
		"""
		return self.get_node(self.parent_id)

	def get_node(self, node_id):
		"""
		Connect with the node's RPC server if not already connected. 
		And then return the connection object
		"""
		
		if self.connection_objs[node_id] is not None:
			return self.connection_objs[node_id]

		elif not self.addresses[node_id]:
			return None

		else:
			self.connection_objs[node_id] = ServerProxy(self.addresses[node_id], allow_none=True)

			while True:
				self.log(self.create_log(CONNECTION,'Waiting for node %s to connect'%(node_id)))
				try:
					self.connection_objs[node_id].receive_message(self.id, CONNECTED)
					self.log(self.create_log(CONNECTION,'Connected with node %s'%(node_id)))
					break
				except Exception as e:
					print(e)
					time.sleep(1)

			return self.connection_objs[node_id]		
	
	### Communication with parent
	def push_to_parent(self, weight_gradient, bias_gradient):
		"""
		Push the gradients to the parent node. Calls parent's RPC function internally
		"""

		if not self.parent_address:
			return

		weight_gradient = [x.tolist() for x in weight_gradient]
		bias_gradient = [x.tolist() for x in bias_gradient]
		data_size = get_size([weight_gradient, bias_gradient])

		self.log(self.create_log(PUSHED,{
									NETWORK_COST : data_size
								}))

		self.simulate_delay(data_size, self.parent_id)
		self.get_parent().push_from_child(weight_gradient, bias_gradient, self.id, self.skiptestdata)

	def pull_from_parent(self):
		"""
		Pull the mode from the parent node. Calls parent's RPC function internally
		"""

		if not self.parent_address:
			return
			
		model = self.get_parent().pull_from_child(self.id)
		model_size = get_size(model)

		model[0] = [np.array(x) for x in model[0]]
		model[1] = [np.array(x) for x in model[1]]

		self.log(self.create_log(PULLED,{
									NETWORK_COST : model_size
								}))
		
		self.simulate_delay(model_size, self.parent_id)
		return model

	def simulate_delay(self, data_size, node_id):
		print("Delay: %f"%(8. * data_size / self.bandwidths[node_id]))
		time.sleep( 8. * data_size / self.bandwidths[node_id] )		

	###-------------------------- RPC functions -----------------------------------------
	def get_loss(self, data):
		"""
		RPC function. Return the loss of the present model at the node
		"""

		return float(self.application.evaluate(data))		
	
	def remote_shutdown(self):
		"""
		Cannot shut down the RPC server from the current thread
		Hence we need to create a separate thread to shut it down
		"""

		t = threading.Thread(target=self.shutdown_thread)
		t.start()

	def shutdown_thread(self):
		"""
		Shuts down RPC server
		"""
		self.server.shutdown()

	def get_update_count(self):
		"""
		Returns the number of updates made to the model
		"""
		return self.policy.updates

	###-------------------------- Meta functions -----------------------------------------
	def send_message(self, receiver_id, msg):
		"""
		Use this function to send the data to whichever node you wish to
		"""

		self.log(self.create_log(CONNECTION,'Sending message to node %d, msg: %s'%(receiver_id, msg)))
		self.get_node(receiver_id).receive_message(self.id, msg)

	def get_parent_update_count(self):
		"""
		Used by AccuracyPolicy
		"""
		count = self.get_parent().get_update_count()
		self.log(self.create_log(CONNECTION,'Got parent update count: %d'%count)) 
		return count

	def get_accuracies(self, skipdata=0):
		"""
		Calculate accuracies for each sensor prediction
		Returns a dictionary of accuracies
		"""

		accuracies = {}
		
		for test_file in self.test_files:
			test_data = self.application.get_test_data(self.test_file_handlers[test_file], self.filesizes[test_file], skipdata)
			accuracies[test_file.split('/')[-1]] = self.application.evaluate(test_data)
		
		return accuracies
		
	def create_log(self, log_type, payload):
		"""
		Predefined format for creating logs
		"""

		log = OrderedDict({
				NODE_ID		: self.id,
				TYPE		: log_type,
				PAYLOAD		: payload,
				TIMESTAMP	: time.time()
			})
		return json.dumps(log)

	def log(self, s):
		"""
		Send log 's' to master
		Write 's' to the log file of this node
		"""

		self.get_master().log_report(s)
		self.log_file.write(s +'\n')
		self.log_file.flush()
