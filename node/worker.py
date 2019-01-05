"""
An implementation of abstract class Node for simulating worker nodes
"""

import os
import csv
import time
import psutil
import threading
from io import StringIO
from collections import OrderedDict

from node.node import *
from utility.const import *

from kafka import KafkaConsumer
from kafka.errors import NoBrokersAvailable

class Worker(Node):

	def __init__(self, data, kafka_server_address):
		super().__init__(data)

		self.sensors = [ str(x) for x in data['sensors'] ]

		self.mini_batch_size = data['mini_batch_size'] 
		self.window_interval = data['window_interval']
		self.window_limit = data['window_limit']
		self.epochs_per_window = data['epochs_per_window']

		self.window_count = 0

		self.skiptestdata = 0

		try:
			self.consumer = KafkaConsumer(bootstrap_servers=str(kafka_server_address), api_version=(0,10))
			self.consumer.subscribe(self.sensors)	
			self.log(self.create_log(CONNECTION,'Connected with the Kafka server'))
		except NoBrokersAvailable:
			print("No Brokers are Available. Please start the Kafka server")
			self.log(self.create_log(CONNECTION,'No Brokers are Available. Please start the Kafka server'))
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
		while self.window_count < self.window_limit:
			epoch_start_cpu, epoch_start = time.process_time(), time.perf_counter()
			data = self.get_data()
			
			self.skiptestdata += len(data)

			### Pull model from parent			
			self.network.use_parent_model(*self.pull_from_parent())

			### Run training algorithm
			self.network.train(data, epochs=self.epochs_per_window, mini_batch_size=self.mini_batch_size)

			### Log Statistics for the epoch
			self.log(self.create_log(STATISTIC,"Window %d processed"%self.window_count))

			self.log(self.create_log(STATISTIC,OrderedDict({
				WINDOW_ID		: self.window_count,
				RUNTIME			: time.perf_counter() - epoch_start,
				PROCESS_TIME	: time.process_time() - epoch_start_cpu,
				MEMORY_USAGE	: py.memory_percent(),
				ACCURACY		: self.get_accuracies(skipdata=self.skiptestdata),
				DATAPOINTS 		: len(data)
			})))

			### Push model to parent
			self.push_to_parent(*self.network.get_and_reset_acquired_gradients())
			self.window_count += 1
		
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

		start = time.time()
		sensor_data_string, data_points = "", 0
		for msg in self.consumer:
			sensor_data_string += msg.value.decode('utf-8')
			data_points+=1
			
			if (time.time() - start) > self.window_interval: break
		
		train_data = list(csv.reader(StringIO(sensor_data_string)))

		return train_data

	###-------------------------- Additional RPC functions ---------------------------------

	def receive_message(self, sender_id, msg):
		"""
		Abstract Method Implementation
		Receive message from other nodes
		"""

		self.log(self.create_log(CONNECTION,'Received message from node id %d: %s'%(sender_id, msg)))   
