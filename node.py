from SimpleXMLRPCServer import SimpleXMLRPCServer
import time
from queue import Queue
import xmlrpclib
import thread
from network import Network



class Node:

	def __init__(self, id, own_address, parent_address):
		self.id = id
		self.own_address = own_address
		self.parent_address = parent_address
		self.connected_with_parent = False

		
		self.acquired_gradients_from_kids = Queue()

		thread.start_new_thread(self.comsume_gradients_from_kids_thread, ())
		thread.start_new_thread(self.run_rpc_server_thread, ())

		
		self.network = Network([1,2])
		self.a = 1
		self.b = 2


	def get_parent(self):
		if self.connected_with_parent:
			return self.parent

		elif not self.parent_address:
			return None
		
		else:
			self.parent = xmlrpclib.ServerProxy(parent_address, allow_none=True)
			
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
			self.network.apply_kid_gradient()


	def run_rpc_server_thread(self):
		server = SimpleXMLRPCServer(self.own_address, allow_none=True)
		server.register_function(self.push_from_child, "push_from_child")
		server.register_function(self.pull_from_child, "pull_from_child")
		server.serve_forever()




parent_address = 'http://localhost:8000'
own_address = ("localhost", 8001)

n1 = Node(1, ("localhost", 8000), None)
n2 = Node(2, own_address, parent_address)

print n1.a
print n2.a
print n2.push_to_parent(2,3)
print n2.pull_from_parent()
print n1.a
print n1.a


time.sleep(10)