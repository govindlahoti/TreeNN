"""
An implementation of abstract class Node for simulating parameter servers
"""

import threading
from queue import Queue

from node.node import *
from utility.const import *

class ParameterServer(Node):

	def __init__(self, data):
		"""
		Extra class variables:
		1. self.acquired_gradients_from_kids: For queueing gradients received from child node
		2. self.merge_id: For maintaing and updating Gradient merges done by Parameter servers atomically
		3. self.active_children and self.child_ever_connected: For stopping simulation after child nodes simulate
		"""

		super().__init__(data) 
		self.acquired_gradients_from_kids = Queue() 

		self.merge_id = 0

		self.active_children = set()
		self.child_ever_connected = False

	def init_threads(self):
		"""
		Abstract Method Implementation
		Parameter server spawns two threads - 
		1. To consume the gradients being obtained from the child node
		2. To run the RPC server
		"""

		consume_thread = threading.Thread(target = self.consume_gradients_from_kids_thread)
		server_thread = threading.Thread(target = self.run_rpc_server_thread)

		consume_thread.start()
		server_thread.start()

		return [consume_thread, server_thread]    

	def consume_gradients_from_kids_thread(self):
		"""
		This thread runs till child nodes stop simulating. 
		In a single rollout:
		1. Monitors the queue of acquired gradients from the child nodes. (get() method is a blocking function call)
		2. Pulls model from parent
		3. Logs the accuracy of the model before merging gradients
		4. Modifies the node's model using those gradients.
		5. Logs the accuracy of the model after merging gradients 
		6. Pushes new model to the parent
		"""

		while len(self.active_children) > 0 or not self.child_ever_connected or not self.acquired_gradients_from_kids.empty():
			weight_gradient, bias_gradient, child_id, skipdata = self.acquired_gradients_from_kids.get()
			
			### Pull from parent after consulting the policy
			if self.policy.pull_from_parent(self):
				self.network.use_parent_model(*self.pull_from_parent())

			# ## Log pre merge accuracy
			# self.log(self.create_log(STATISTIC,OrderedDict({
			# 		MERGE_ID 			: self.merge_id,
			# 		PRE_MERGE_ACCURACY	: self.get_accuracies()
			# 	})))

			### Merge gradients
			self.network.apply_kid_gradient(weight_gradient, bias_gradient)
			self.policy.updates += 1
			
			### Log post merge accuracy
			self.log(self.create_log(MERGED, 'Merged gradients at node %d'%(self.id)))

			self.accuracies = self.get_accuracies(skipdata=skipdata)
			self.log(self.create_log(STATISTIC, OrderedDict({
					MERGE_ID			: self.merge_id,
					CHILD_ID 			: child_id,
					SKIP_TEST_DATA 		: skipdata,
					POST_MERGE_ACCURACY	: self.accuracies
				})))	
			self.skiptestdata = max(self.skiptestdata,skipdata)

			### Push Gradient to parent after consulting the policy
			if self.policy.push_to_parent(self):
				self.push_to_parent(*self.network.get_and_reset_acquired_gradients())

			self.merge_id += 1

		### Stop Server Thread
		client = ServerProxy(self.own_server_address)
		client.remote_shutdown()

		### Send Done acknowledgement to Master
		self.log(self.create_log(DONE, ''))

	def run_rpc_server_thread(self):
		"""
		Abstract Method Implementation
		Thread to run the RPC server for the node
		"""

		self.server = SimpleXMLRPCServer(self.own_tuple_address, allow_none=True)
		self.server.register_function(self.push_from_child, "push_from_child")
		self.server.register_function(self.pull_from_child, "pull_from_child")
		self.server.register_function(self.get_loss, "get_loss")
		self.server.register_function(self.receive_message, "receive_message")
		self.server.register_function(self.remote_shutdown, "remote_shutdown")
		self.server.register_function(self.get_update_count, "get_update_count")
		# add functions for communication between workers
		self.server.serve_forever()

	###-------------------------- Additional RPC functions ---------------------------------

	def push_from_child(self, weight_gradient, bias_gradient, child_id, skipdata):
		"""
		RPC function. Add the gradients obtained from child node into the queue
		"""

		self.log(self.create_log(CONNECTION, 'Got gradients from child %d'%(child_id)))
		
		weight_gradient = [np.array(x)/len(self.active_children) for x in weight_gradient]
		bias_gradient = [np.array(x)/len(self.active_children) for x in bias_gradient]

		print(child_id, skipdata)
		self.acquired_gradients_from_kids.put([weight_gradient, bias_gradient, child_id, skipdata])

	def pull_from_child(self, child_id):
		"""
		RPC function. Return the model (weights and biases) to the child node
		"""

		self.log(self.create_log(CONNECTION,'Got pull request from child %d'%(child_id)))
		model = self.network.get_model()
		model[0] = [x.tolist() for x in model[0]]
		model[1] = [x.tolist() for x in model[1]]
		return model

	def receive_message(self, sender_id, msg):
		"""
		Abstract Method Implementation
		RPC function called by child nodes. Keep track of the child's current status
		"""

		self.child_ever_connected = True
		if msg == CONNECTED: 
			self.active_children.add(sender_id)
		elif msg == DISCONNECTED: 
			self.active_children.remove(sender_id)

		self.log(self.create_log(CONNECTION,'Received message from node id %d: %s'%(sender_id, msg)))   
