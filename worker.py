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
		server_thread = threading.Thread(target = self.run_rpc_server_thread)
		logic_thread = threading.Thread(target = self.run_sharing_logic_thread)

		server_thread.start()
		logic_thread.start()

		return [logic_thread, server_thread]    

	def run_rpc_server_thread(self):
		"""Thread to run the RPC server for the node"""
		self.server = SimpleXMLRPCServer(self.own_tuple_address, allow_none=True)
		self.server.register_function(self.get_loss, "get_loss")
		self.server.register_function(self.recv_message, "recv_message")
		self.server.register_function(self.remote_shutdown, "remote_shutdown")
		# add functions for communication between workers
		self.server.serve_forever()

	def run_sharing_logic_thread(self):

		while self.e <= self.epoch_limit:
			data = get_data(self.inputfile)
			
			### Pull model from parent			
			self.network.use_parent_model(*self.pull_from_parent())

			### Run SGD algorithm
			self.network.SGD(data, epochs=1, mini_batch_size=self.batch_size)

			self.log(self.create_log('STAT',OrderedDict({
				'Epoch ID': self.e,
				'Runtime': time.time(),
				'Accuracy':self.network.evaluate(data),
				})))

			### Push model to parent
			self.push_to_parent(*self.network.get_and_reset_acquired_gradients())
			
			self.e += 1
			
		client = ServerProxy(self.own_server_address)
		client.remote_shutdown()