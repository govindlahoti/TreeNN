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
from io import StringIO
from collections import OrderedDict
from distlib.util import CSVReader

from utility.const import *
from app.network import *

from abc import ABC, abstractmethod

from xmlrpc.client import ServerProxy
from xmlrpc.server import SimpleXMLRPCServer

def get_data(filename):
	"""
	read data of sensors from file
	"""

	with open(filename) as csv_file:
		print("reading data")
		csv_reader = csv.reader(csv_file)
		train_data = list(csv_reader)
		# print(train_data[0])
		print("train data", len(train_data))
	
	return train_data

def get_size(l):
	"""
	Returns the size of list/tuple (even if it is nested)
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
		self.parent_id = data['parent_id']
		self.parent_address = data['parent_address']
		
		### Information about connections 
		self.connection_objs = {}
		self.addresses = { int(k):v for k,v in data['addresses'].items() }
		for w in data['delays']:
			self.connection_objs[int(w)] = None
		self.delays = { int(k):v for k,v in data['delays'].items() }

		### Information about the learning model
		test_files = os.listdir(data['test_directory'])
		self.test_files = [open(data['test_directory']+f, 'r') for f in test_files]
		self.network = Network([276, 276, 276, 48])

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

		time.sleep(self.delays[node_id] / 1000.)
		
		if self.connection_objs[node_id] is not None:
			return self.connection_objs[node_id]

		elif not self.addresses[node_id]:
			return None

		else:
			self.connection_objs[node_id] = ServerProxy(self.addresses[node_id], allow_none=True)
			print(self.addresses[node_id],self.connection_objs[node_id])
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

		self.log(self.create_log(CONNECTION,'Sending gradients to parent %d'%(self.parent_id)))
		weight_gradient = [x.tolist() for x in weight_gradient]
		bias_gradient = [x.tolist() for x in bias_gradient]

		self.log(self.create_log(CONNECTION,{
									NETWORK_COST : get_size([weight_gradient, bias_gradient])
								}))

		self.get_parent().push_from_child(weight_gradient, bias_gradient, self.id)

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

		self.log(self.create_log(CONNECTION,'Got model from parent %d'%(self.parent_id)))
		self.log(self.create_log(CONNECTION,{
									NETWORK_COST : model_size
								}))
		
		return model

	###-------------------------- RPC functions -----------------------------------------
	def get_loss(self):
		"""
		RPC function. Return the loss of the present model at the node
		"""

		return float(self.network.evaluate(data))		
	
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

	###-------------------------- Meta functions -----------------------------------------
	def send_message(self, receiver_id, msg):
		"""
		Use this function to send the data to whichever node you wish to
		"""

		self.log(self.create_log(CONNECTION,'Sending message to node %d, msg: %s'%(receiver_id, msg)))
		self.get_node(receiver_id).receive_message(self.id, msg)

	def get_accuracies(self):
		"""
			Calculate accuracies for each sensor prediction
			Returns a dictionary of accuracies
		"""

		accuracies = {}
		
		for test_file in self.test_files:
			test_data = self.get_test_data(test_file)
			accuracies[test_file.name.split('/')[-1]] = self.network.evaluate(test_data)
		
		return accuracies

	def get_test_data(self, test_file, size=50):
		"""
			Get test data from file. 
			Using random-seeking in file to limit RAM usage
		"""		

		sample = []
		test_file.seek(0, 2)
		filesize = test_file.tell()

		random_set = sorted(random.sample(range(filesize), size))

		for i in range(size):
			test_file.seek(random_set[i])
			# Skip current line (because we might be in the middle of a line) 
			test_file.readline()
			# Append the next line to the sample set 
			sample.append(test_file.readline())

		test_data = list(csv.reader(StringIO("".join(sample))))

		return test_data
		
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
