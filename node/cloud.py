"""
An implementation of abstract class Node for simulating the Cloud 
"""

import os
import csv
import time
import psutil
import threading
from collections import OrderedDict

from node.node import *
from utility.const import *

from kafka import KafkaConsumer
from kafka.errors import NoBrokersAvailable

class Cloud(Node):

	def __init__(self, data, kafka_server_address):
		super().__init__(data)

		self.id = -1
		self.sensors = [ str(x) for x in data['sensors'] ]

		self.window_interval = data['window_interval']
		self.window_limit = data['window_limit']
		self.window_count = 0

		self.data_collection_start = None

		# For proper cleaning up
		self.active_connections = set()
		self.dead_connections = set()

		self.done = False

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
		Cloud spawns two threads - 
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
		self.server.register_function(self.get_update_count, "get_update_count")
		# add functions for communication between Clouds
		self.server.serve_forever()

	def training_thread(self):
		"""
		Runs for the number of epochs specifies in the configuration file.
		For each epoch:
		1. Runs training algorithm as defined by the application
		2. Log Statistics for the epoch:
			a. Epoch ID
			b. Runtime
			c. Process time
			d. Memory usage recorded at the end of the epoch
			e. Accuracy of the model
		"""
		
		py = psutil.Process(os.getpid())
		print(time.time(), ": Called RPC")
		self.log(self.create_log(TRAINING, 'Ready to start training'))
		condition = self.log(self.create_log(QUERY, 'Checking whether hierarchy is established'))
		while (not condition):
			time.sleep(1)
			condition = self.log(self.create_log(QUERY, 'Checking whether hierarchy is established'))
		
		print(time.time(), ": Started Training")

		self.data_collection_start = time.time()
		while self.window_count < self.window_limit:
			print(GREENSTR%str(str(time.strftime("%H:%M:%S", time.localtime(time.time()))) + " Processing window: " + str(self.window_count)))
			epoch_start_cpu, epoch_start = time.process_time(), time.perf_counter()
			print("Called Get data")
			data = self.application.transform_sensor_data(self.get_data(max(time.time(), self.data_collection_start + self.window_interval)))
			self.data_collection_start = time.time()
			print("Received " + str(len(data)))			
			self.skiptestdata += len(data)

			### Run training algorithm
			self.application.train(data)

			### Log Statistics for the epoch
			self.log(self.create_log(PROCESSED, { 
				WINDOW_ID		: self.window_count,
				MEMORY_USAGE	: py.memory_percent(), 
				}))

			self.accuracies = self.get_accuracies(skipdata=self.skiptestdata)

			self.log(self.create_log(STATISTIC,OrderedDict({
				WINDOW_ID		: self.window_count,
				RUNTIME			: time.perf_counter() - epoch_start,
				PROCESS_TIME	: time.process_time() - epoch_start_cpu,
				ACCURACY		: self.accuracies,
				DATAPOINTS 		: len(data)
			})))

			self.window_count += 1

		while self.done != True:
			print("Sleeping")
			time.sleep(10)
		self.cleanup()
		
	def get_data(self, last_timestamp):
		"""
		Consumes data points received from Kafka in batches
		"""

		sensor_data_string, data_points = "", 0
		for msg in self.consumer:
			sensor_data_string += msg.value.decode('utf-8')
			data_points+=1
			
			if msg.timestamp//1000 > last_timestamp: break
		
		return sensor_data_string

	def cleanup(self):

		print("Completed the job. Exiting")
		
		### Stop Server Thread
		client = ServerProxy(self.own_server_address)
		client.remote_shutdown()

		### Send Done acknowledgement to Master
		self.log(self.create_log(DONE, ''))

	###-------------------------- Additional RPC functions ---------------------------------

	def receive_message(self, sender_id, msg):
		"""
		Abstract Method Implementation
		Receive message from other nodes
		"""

		if msg == CONNECTED: 
			self.active_connections.add(sender_id)
		elif msg == DISCONNECTED:
			print("Received DISCONNECTED message from ", sender_id)
			self.dead_connections.add(sender_id)
			print("Dead Connections %d/%d" % (len(self.dead_connections), len(self.active_connections)))

			if len(self.active_connections) == len(self.dead_connections):
				self.done = True

		else:
			print(time.time(), "Receive Message (ping cloud) by ", sender_id)
			## Test accuracy. Message = Skipdata
			self.accuracies = self.get_accuracies(skipdata=int(msg))

			self.log(self.create_log(CLSTATISTIC,OrderedDict({
				NODE_ID			: sender_id,
				ACCURACY		: self.accuracies,
			})))

			# print(time.time(), "Receive Message Finished (ping cloud) by ", sender_id)

		self.log(self.create_log(CONNECTION,'Received message from node id %d: %s'%(sender_id, msg)))