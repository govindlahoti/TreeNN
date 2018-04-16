from SimpleXMLRPCServer import SimpleXMLRPCServer
import time
from queue import Queue
import xmlrpclib
import thread
from network import *



class Node:

	def __init__(self, id, own_address, parent_address, is_worker):
		self.id = id
		self.own_address = own_address
		self.parent_address = parent_address
		self.is_worker = is_worker
		self.connected_with_parent = False
		
		self.acquired_gradients_from_kids = Queue()

		thread.start_new_thread(self.comsume_gradients_from_kids_thread, ())
		thread.start_new_thread(self.run_rpc_server_thread, ())
		thread.start_new_thread(self.run_sharing_logic_thread, ())

		self.run_sharing_logic()

	def get_parent(self):
		if self.connected_with_parent:
			return self.parent

		elif not self.parent_address:
			return None
		
		else:
			self.parent = xmlrpclib.ServerProxy(self.parent_address, allow_none=True)
			
			while True:
				print 'Waiting for parent', self.parent_address, ' ...'
				try:
					self.parent.pull_from_child()
					print 'Connected with parent', self.parent_address
					self.connected_with_parent = True
					break
				except:
					time.sleep(1)

			return self.parent


	def push_from_child(self, weight_gradient, bias_gradient):
		self.acquired_gradients_from_kids.put((weight_gradient, bias_gradient))
	

	def pull_from_child(self):
		return self.network.get_model()
	

	def push_to_parent(self, weight_gradient, bias_gradient):
		if not self.parent_address:
			return
		self.get_parent().push_from_child(weight_gradient, bias_gradient)


	def pull_from_parent(self):
		if not self.parent_address:
			return
		return self.get_parent().pull_from_child()


	def comsume_gradients_from_kids_thread(self):
		while True:
			weight_gradient, bias_gradient = self.acquired_gradients_from_kids.get()
			self.network.apply_kid_gradient(weight_gradient, bias_gradient)


	def run_rpc_server_thread(self):
		server = SimpleXMLRPCServer(self.own_address, allow_none=True)
		server.register_function(self.push_from_child, "push_from_child")
		server.register_function(self.pull_from_child, "pull_from_child")
		server.serve_forever()

	def run_sharing_logic_thread(self):
		if self.is_worker:
			self.network = Network([1,2])

			for e in range(epochs):
				if e % worker_fetch_interval == 0:
					if self.parent_address:
						self.network.use_parent_model(*self.pull_from_parent())

				self.network.SGD(self.get_data())

				if e % worker_push_interval == 0:
					if self.parent_address:
						self.push_to_parent(*self.network.get_and_reset_acquired_gradients())
		else:
			while True:
				if e % worker_fetch_interval == 0:
					if self.parent_address:
						self.network.use_parent_model(*self.pull_from_parent())

				if e % worker_push_interval == 0:
					if self.parent_address:
						self.push_to_parent(*self.network.get_and_reset_acquired_gradients())

