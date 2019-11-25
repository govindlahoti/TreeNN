"""
An implementation of abstract class Node for simulating worker nodes
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

class Worker(Node):

	def __init__(self, data, kafka_server_address):
		super().__init__(data)
		print("I have started")

		self.sensors = [ str(x) for x in data['sensors'] ]

		self.window_interval = data['window_interval']
		self.window_limit = data['window_limit']
		self.window_count = 0

		self.data_collection_start = None

		# Specific for Synchronous learning
		self.start = False
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
		Worker spawns two threads - 
		1. To run the training thread
		2. To run the RPC server
		"""
		train_thread = None
		if self.learning == "Synchronous":
			train_thread = threading.Thread(target = self.synchronous)
		elif self.learning == "Asynchronous":
			train_thread = threading.Thread(target = self.asynchronous)
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
		# add functions for communication between workers
		self.server.serve_forever()

	def synchronous(self):
		py = psutil.Process(os.getpid())
		print(time.time(), ": Called RPC")
		self.log(self.create_log(TRAINING, 'Ready to start training'))
		condition = self.log(self.create_log(QUERY, 'Checking whether hierarchy is established'))
		while (not condition):
			time.sleep(1)
			condition = self.log(self.create_log(QUERY, 'Checking whether hierarchy is established'))
		
		print(time.time(), ": Started")

		while True:
			epoch_start_cpu, epoch_start = time.process_time(), time.perf_counter()
			print("Called Get data")
			data = self.application.transform_sensor_data(self.get_data())
			print("Checking done condition")
			if self.done:
				print("Exiting Training thread")
				self.cleanup("Synchronous")
				break
			print("Received " + str(len(data)))
			self.skiptestdata += len(data)

			### Pull model from parent after consulting the policy			
			print("Puling Model from parent")
			if self.policy.pull_from_parent(self):
				self.application.use_parent_model(*self.pull_from_parent())

			# TODO : Remove pre merging accuracies, Validating that parent is transferring model
			print("Pre Training accuracies")
			self.accuracies = self.get_accuracies(skipdata=self.skiptestdata, cloud = False)
			self.log(self.create_log(STATISTIC,OrderedDict({
				WINDOW_ID		: self.window_count,
				RUNTIME			: time.perf_counter() - epoch_start,
				PROCESS_TIME	: time.process_time() - epoch_start_cpu,
				ACCURACY		: self.accuracies,
				DATAPOINTS 		: len(data)
			})))


			### Run training algorithm
			print("Performing training")
			self.application.train(data)

			### Log Statistics for the epoch
			self.log(self.create_log(PROCESSED, { 
				WINDOW_ID		: self.window_count,
				MEMORY_USAGE	: py.memory_percent(), 
				}))

			print("Post Training accuracies")
			self.accuracies = self.get_accuracies(skipdata=self.skiptestdata)

			self.log(self.create_log(STATISTIC,OrderedDict({
				WINDOW_ID		: self.window_count,
				RUNTIME			: time.perf_counter() - epoch_start,
				PROCESS_TIME	: time.process_time() - epoch_start_cpu,
				ACCURACY		: self.accuracies,
				DATAPOINTS 		: len(data)
			})))

			### Push model to parent after consulting the policy
			print("Pushing model to parent")
			if self.policy.push_to_parent(self):
				self.push_to_parent(*self.application.get_and_reset_acquired_gradients())

			self.send_message(self.parent_id, FINISH)
			self.start = False
			time.sleep(1)

	def asynchronous(self):
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
			data = self.application.transform_sensor_data(self.get_data(max(time.time(), self.data_collection_start+self.window_interval)))
			self.data_collection_start = time.time()
			print("Received " + str(len(data)))
			self.skiptestdata += len(data)

			### Pull model from parent after consulting the policy			
			print("Puling Model from parent")
			if self.policy.pull_from_parent(self):
				self.application.use_parent_model(*self.pull_from_parent())

			### Run training algorithm
			print("Performing training")
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

			### Push model to parent after consulting the policy
			print("Pushing model to parent")
			if self.policy.push_to_parent(self):
				self.push_to_parent(*self.application.get_and_reset_acquired_gradients())

			self.window_count += 1
		
		time.sleep(self.window_interval)
		self.cleanup()

	def get_data(self, last_timestamp=-1):
		"""
		Consumes data points received from Kafka in batches
		"""

		sensor_data_string, data_points = "", 0
		for msg in self.consumer:
			sensor_data_string += msg.value.decode('utf-8')
			data_points+=1
			
			if last_timestamp == -1:
				if (self.start or self.done):	break
			elif msg.timestamp//1000 > last_timestamp:	break
		
		return sensor_data_string

	def cleanup(self, type="Asynchronous"):
		print("Cleaning up")
		if type == "Asynchronous":
			### Alert parent that training is done
			self.send_message(self.parent_id, DISCONNECTED)

		if self.cloud_exists:
			self.ping_cloud(DISCONNECTED)

		print("Completed the job. Exiting")
		
		### Stop server thread
		client = ServerProxy(self.own_server_address)
		print("Shutting down the server")
		client.remote_shutdown()
		print("Shutting down the server")
		
		### Send Done acknowledgement to Master
		self.log(self.create_log(DONE,''))

		print("Cleanup Finished")


	###-------------------------- Additional RPC functions ---------------------------------

	def receive_message(self, sender_id, msg):
		"""
		Abstract Method Implementation
		Receive message from other nodes
		"""

		self.log(self.create_log(CONNECTION,'Received message from node id %d: %s'%(sender_id, msg)))

		if msg == START:
			self.start = True

		elif msg == DONE:
			print("Received DONE Signal")
			self.done = True