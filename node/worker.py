"""
An implementation of abstract class Node for simulating worker nodes
"""

import os
import csv
import psutil
import threading
from collections import OrderedDict

from node.node import *
from utility.const import *

from kafka import KafkaConsumer
from kafka.errors import NoBrokersAvailable

class Worker(Node):

	def __init__(self, data):
		super().__init__(data)
		
		self.window_size = data['window_size']
		self.mini_batch_size = data['mini_batch_size'] 

		self.epoch_limit = data['epoch_limit']
		self.epoch_count = 0

		try:
			self.consumer = KafkaConsumer(str(self.id))
		except NoBrokersAvailable:
			print("No Brokers are Available. Please start the Kafka server")
			exit(0)
			
	def init_threads(self):
		"""
		Abstract Method Implementation
		Worker spawns two threads - 
		1. To run the training thread
		2. To run the RPC server
		"""
		train_thread = threading.Thread(target = self.training_thread)
		server_thread = threading.Thread(target = self.run_rpc_server_thread)
		
		train_thread.start()
		server_thread.start()

		return [train_thread, server_thread] 

	def run_rpc_server_thread(self):
		"""
		Abstract Method Implementation
		Thread to run the RPC server for the node
		"""

		self.server = SimpleXMLRPCServer(self.own_tuple_address, allow_none=True)
		self.server.register_function(self.get_loss, "get_loss")
		self.server.register_function(self.receive_message, "receive_message")
		self.server.register_function(self.remote_shutdown, "remote_shutdown")
		# add functions for communication between workers
		self.server.serve_forever()

	def training_thread(self):
		"""
		Runs for the number of epochs specifies in the configuration file.
		For each epoch:
		1. Pulls model from parent 
		2. Runs training algorithm as defined by the application
		3. Log Statistics for the epoch:
			a. Epoch ID
			b. Runtime
			c. Process time
			d. Memory usage recorded at the end of the epoch
			e. Accuracy of the model
		4. Push model to the parent
		"""
		
		py = psutil.Process(os.getpid())
		while self.epoch_count < self.epoch_limit:
			epoch_start_cpu, epoch_start = time.time(), time.clock()
			data = self.get_data()
			
			### Pull model from parent			
			self.network.use_parent_model(*self.pull_from_parent())

			### Run training algorithm
			self.network.train(data, epochs=1, mini_batch_size=self.mini_batch_size)

			### Log Statistics for the epoch
			self.log(self.create_log(STATISTIC,OrderedDict({
				'Epoch ID'		: self.epoch_count,
				'Runtime'		: time.clock() - epoch_start,
				'Process time'	: time.time() - epoch_start_cpu,
				'Memory Usage'	: py.memory_info()[0]/2.**30,
				'Accuracy'		: self.network.evaluate(data),
				})))

			### Push model to parent
			self.push_to_parent(*self.network.get_and_reset_acquired_gradients())
			self.epoch_count += 1
		
		### Alert parent that training is done
		self.send_message(self.parent_id,DISCONNECTED)

		### Stop server thread
		client = ServerProxy(self.own_server_address)
		client.remote_shutdown()

		### Send Done acknowledgement to Master
		self.log(self.create_log(DONE,''))

	def get_data(self):
		"""
			Consumes data points received from Kafka in batches
		"""
		sensor_data, data_points = [], 0
		for msg in self.consumer:
			# if data_points==0: print([msg.value.decode('utf-8')[1:-1].replace("\\", "")])
			sensor_data.append(csv.reader(msg.value.decode('utf-8')))
			data_points+=1
			if data_points == self.window_size:
				print(data_points,self.window_size)
				break
		print(sensor_data[0])
		return sensor_data
		

	###-------------------------- Additional RPC functions ---------------------------------

	def receive_message(self, sender_id, msg):
		"""
		Abstract Method Implementation
		Receive message from other nodes
		"""

		self.log(self.create_log(CONNECTION,'Received message from node id %d: %s'%(sender_id, msg)))   
