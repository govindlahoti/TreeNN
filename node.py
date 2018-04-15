from SimpleXMLRPCServer import SimpleXMLRPCServer
import time

# def is_even(n):
# 	time.sleep(10)
# 	return n % 2 == 0

# server = SimpleXMLRPCServer(("localhost", 8000))
# print("Listening on port 8000...")
# server.register_function(is_even, "is_even")
# server.serve_forever()
# print 'ok'

from queue import Queue
import xmlrpclib
import thread
# import network

class Node:

	def __init__(self, id, own_address, parent_address):
		self.id = id
		self.own_address = own_address
		self.parent_address = parent_address

		if parent_address:
			self.parent = xmlrpclib.ServerProxy(parent_address, allow_none=True)
		
		self.acquired_gradients = Queue()

		thread.start_new_thread(self.run_rpc_server, ())

		
		self.a = 1
		self.b = 2

	def push_from_child(self, weight_gradient, bias_gradient):
		self.a = weight_gradient
		self.b = bias_gradient
	
	def pull_from_child(self):
		return self.a, self.b
	
	def push_to_parent(self, weight_gradient, bias_gradient):
		if not self.parent_address:
			return
		self.parent.push_from_child(weight_gradient, bias_gradient)


	def pull_from_parent(self):
		if not self.parent_address:
			return
		return self.parent.pull_from_child()


	def run_rpc_server(self):
		server = SimpleXMLRPCServer(self.own_address, allow_none=True)
		server.register_function(self.push_from_child, "push_from_child")
		server.register_function(self.pull_from_child, "pull_from_child")
		server.serve_forever()




parent_address = 'http://localhost:8000'
own_address = ("localhost", 8001)

n1 = Node(1, ("localhost", 8000), None)
time.sleep(2)
n2 = Node(2, own_address, parent_address)

print n1.a
print n2.a
print n2.push_to_parent(2,3)
print n2.pull_from_parent()
print n1.a
print n2.a


time.sleep(10)