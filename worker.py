import os
import psutil
import threading
from collections import OrderedDict

from node import *

class Worker(Node):

	def __init__(self, data):
		super().__init__(data) 
		self.e = 0

	def init_threads(self):
		'''
		Worker spawns two threads - 
		1. To run the downpour SGD algorithm
		2. To run the RPC server
		'''
		train_thread = threading.Thread(target = self.training_thread)
		server_thread = threading.Thread(target = self.run_rpc_server_thread)

		train_thread.start()
		server_thread.start()

		return [train_thread, server_thread] 

	def run_rpc_server_thread(self):
		"""Thread to run the RPC server for the node"""
		self.server = SimpleXMLRPCServer(self.own_tuple_address, allow_none=True)
		self.server.register_function(self.get_loss, "get_loss")
		self.server.register_function(self.receive_message, "receive_message")
		self.server.register_function(self.remote_shutdown, "remote_shutdown")
		# add functions for communication between workers
		self.server.serve_forever()

	def training_thread(self):

		py = psutil.Process(os.getpid())
		while self.e < self.epoch_limit:
			epoch_start_cpu, epoch_start = time.time(), time.clock()
			data = get_data(self.inputfile)
			### Pull model from parent			
			self.network.use_parent_model(*self.pull_from_parent())

			### Run SGD algorithm
			self.network.SGD(data, epochs=1, mini_batch_size=self.batch_size)

			self.log(self.create_log('STAT',OrderedDict({
				'Epoch ID': self.e,
				'Runtime': time.clock() - epoch_start,
				'Process time': time.time() - epoch_start_cpu,
				'Memory Usage': py.memory_info()[0]/2.**30,
				'Accuracy':self.network.evaluate(data),
				})))

			### Push model to parent
			self.push_to_parent(*self.network.get_and_reset_acquired_gradients())
			self.e += 1
		
		### Alert parent that training is done
		self.send_message(self.parent_id,'disconnected')

		### Stop server thread
		client = ServerProxy(self.own_server_address)
		client.remote_shutdown()

		### Send Done acknowledgement to Master
		self.log(self.create_log('DONE',''))

	###-------------------------- Additional RPC functions ---------------------------------

	def receive_message(self, sender_id, msg):
		self.log(self.create_log('CONN','Received message from node id %d: %s'%(sender_id, msg)))   
