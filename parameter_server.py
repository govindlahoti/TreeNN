import threading
from queue import Queue

from node import *

class ParameterServer(Node):

	def __init__(self, data):
		super().__init__(data) 
		self.acquired_gradients_from_kids = Queue() 
		self.data = get_data(self.inputfile)
		self.merge_id = 0
		self.merge_lock = threading.Lock()

	def init_threads(self):
		'''
		Parameter server spawns three threads - 
		1. To consume the gradients being obtained from the child node
		2. To run the downpour SGD algoritm (sharing logic)
		3. To run the RPC server
		'''
		consume_thread = threading.Thread(target = self.comsume_gradients_from_kids_thread)
		server_thread = threading.Thread(target = self.run_rpc_server_thread)

		consume_thread.start()
		server_thread.start()

		return [consume_thread, server_thread]    

	def comsume_gradients_from_kids_thread(self):
		"""
		This thread runs indefinitely. It monitors the queue of acquired gradients from the 
		child nodes. It modifies the node's model using those gradients.
		"""
		while True:
			weight_gradient, bias_gradient = self.acquired_gradients_from_kids.get()
			
			### Pull from parent
			if self.parent_address:
				self.network.use_parent_model(*self.pull_from_parent())

			### Log pre merge accuracy
			self.log(self.create_log('STAT',OrderedDict({
					'Merge ID': self.merge_id,
					'Pre Merge Accuracy':self.network.evaluate(self.data),
					})))

			self.network.apply_kid_gradient(weight_gradient, bias_gradient)

			### Push to parent
			self.log(self.create_log('CONN','Merged gradients at node %d'%(self.id)))
		
			### Log pre merge accuracy
			self.log(self.create_log('STAT',OrderedDict({
					'Merge ID': self.merge_id,
					'Post Merge Accuracy':self.network.evaluate(self.data),
					})))	

			### Push Gradient to parent
			if self.parent_address:
				self.push_to_parent(*self.network.get_and_reset_acquired_gradients())

			self.merge_lock.acquire()
			self.merge_id += 1
			self.merge_lock.release()

		client = ServerProxy(self.own_server_address)
		client.remote_shutdown()

	def run_rpc_server_thread(self):
		"""Thread to run the RPC server for the node"""
		self.server = SimpleXMLRPCServer(self.own_tuple_address, allow_none=True)
		self.server.register_function(self.push_from_child, "push_from_child")
		self.server.register_function(self.pull_from_child, "pull_from_child")
		self.server.register_function(self.get_loss, "get_loss")
		self.server.register_function(self.recv_message, "recv_message")
		self.server.register_function(self.remote_shutdown, "remote_shutdown")
		# add functions for communication between workers
		self.server.serve_forever()

	###-------------------------- RPC functions -----------------------------------------

	def push_from_child(self, weight_gradient, bias_gradient, child_id):
		"""RPC function. Add the gradients obtained from child node into the queue"""
		self.log(self.create_log('CONN','Got gradients from child %d'%(child_id)))
		
		if(str(child_id)=='2' or str(str(child_id) == '3')):
			weight_gradient = [np.array(x)/2 for x in weight_gradient]
			bias_gradient = [np.array(x)/2 for x in bias_gradient]
			self.log(self.create_log('CONN','from child1 ' + str(child_id)))
		else:
			weight_gradient = [np.array(x)/3 for x in weight_gradient]
			bias_gradient = [np.array(x)/3 for x in bias_gradient]
			self.log(self.create_log('CONN','from child2 ' + str(child_id)))

		self.acquired_gradients_from_kids.put([weight_gradient, bias_gradient])

	def pull_from_child(self, child_id):
		"""RPC function. Return the model (weights and biases) to the child node"""
		self.log(self.create_log('CONN','Got pull request from child %d'%(child_id)))
		model = self.network.get_model()
		model[0] = [x.tolist() for x in model[0]]
		model[1] = [x.tolist() for x in model[1]]
		return model