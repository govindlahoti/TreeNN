"""
An implementation of abstract class Node for simulating parameter servers
"""

import threading
import numpy as np
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

		self.done = False
		# Specific for Synchronous learning
		self.window_interval = data['window_interval']
		self.window_limit = data['window_limit']
		self.window_count = 0
		self.last_window_time = None
		self.start = False
		self.finish_pending = 0
		self.finish_pending_lock = threading.Lock()

	def init_threads(self):
		"""
		Abstract Method Implementation
		Parameter server spawns two threads - 
		1. To consume the gradients being obtained from the child node
		2. To run the RPC server
		"""

		consume_thread = None
		if self.learning == "Synchronous":
			consume_thread = threading.Thread(target = self.synchronous)
		elif self.learning == "Asynchronous":
			consume_thread = threading.Thread(target = self.asynchronous)
		server_thread = threading.Thread(target = self.run_rpc_server_thread)

		consume_thread.start()
		server_thread.start()

		return [consume_thread, server_thread]    

	def trainSubtree(self):
		'''
		This function is specifically for Synchronous Learning
		It does as follows for the given node
		1)	Signal children to start learning
		2)	Wait untill all children are finished
		3)	Updates its model & logs accuracy
		'''

		### Signal children to start learning
		print(time.time(), "Sending START signal")
		self.finish_pending_lock.acquire()
		for child in self.children:
			self.send_message(child, START)
			self.finish_pending += 1
		self.finish_pending_lock.release()
					
		###	Wait untill all children are finished
		print("Waiting for FINISH signal")
		while (self.finish_pending != 0):
			time.sleep(1)
					
		### Updates its model & logs accuracy
		print("Updating model")
		while not self.acquired_gradients_from_kids.empty():
			weight_gradient, bias_gradient, child_id, skipdata = self.acquired_gradients_from_kids.get()
			self.application.apply_kid_gradient(weight_gradient, bias_gradient)
		
			self.log(self.create_log(MERGED, 'Merged gradients at node %d'%(self.id)))
			self.merge_id += 1
		
		print("Post Merging accuracies")
		self.accuracies = self.get_accuracies()
		self.log(self.create_log(STATISTIC, OrderedDict({
				MERGE_ID			: self.merge_id,
				POST_MERGE_ACCURACY	: self.accuracies
			})))
		# self.skiptestdata = max(self.skiptestdata,skipdata)
		self.policy.updates += 1

	def synchronous(self):
		"""
		This thread runs forever
		As START signal reaches from parent, it does the following:
		1. It pulls model from parent according to its policy
		2. Send START signal to its children
		3. Once all children have send their FINISH signal, It modifies its model using collected gradients and Logs accuracy
		4. It pushes model according to policy and send a FINISH signal to its parent
		"""
		## Waiting for hierarchy to get established
		condition = self.log(self.create_log(QUERY, 'Checking whether hierarchy is established'))
		while (not condition):
			time.sleep(1)
			condition = self.log(self.create_log(QUERY, 'Checking whether hierarchy is established'))

		print(time.time(), "Started")
		## Root Node
		if self.parent_id == -1:
			# It is the root node
			# It doesn't care about self.start 
			# but checks if time since last sending of START signal is more than window_interval or not
			self.last_window_time = time.time()
			while self.window_count < self.window_limit:
				if (time.time() - self.last_window_time) > self.window_interval:
					print(GREENSTR%str(str(time.strftime("%H:%M:%S", time.localtime(time.time()))) + " Processing window: " + str(self.window_count)))
					self.last_window_time = time.time()
					self.window_count += 1
					self.trainSubtree()

				else:
					time.sleep(1)

			print("Sending DONE Signal in " + str(self.window_interval))
			time.sleep(self.window_interval)
			for child in self.children:
				print("Sending DONE Signal to child ", child)
				self.send_message(child, DONE)
			self.cleanup("Synchronous")
		else:
			while (True):
				if (self.start):
					print("Received START signal")
					### Pull from parent after consulting the policy
					if self.policy.pull_from_parent(self):
						self.application.use_parent_model(*self.pull_from_parent())

					self.trainSubtree()

					### Push Gradient to parent after consulting the policy
					if self.policy.push_to_parent(self):
						self.push_to_parent(*self.application.get_and_reset_acquired_gradients())

					### Send FINISH to parent
					print("Sending FINISH signal")
					self.send_message(self.parent_id, FINISH)

					self.start = False
					print("Waiting to start")
				if self.done:
					self.cleanup("Synchronous")
					break
				time.sleep(1)

	def asynchronous(self):
		"""
		This thread runs till child nodes stop simulating. 
		In a single rollout:
		1. Monitors the queue of acquired gradients from the child nodes. (get() method is a blocking function call)
		2. Pulls model from parent
		# 3. Logs the accuracy of the model before merging gradients
		4. Modifies the node's model using those gradients.
		5. Logs the accuracy of the model after merging gradients 
		6. Pushes new model to the parent
		"""

		# while len(self.active_children) > 0 or not self.child_ever_connected or not self.acquired_gradients_from_kids.empty():
		while True:
			# print(list(self.acquired_gradients_from_kids.queue))
			print("Merging gradients from queue %s"% str(self.merge_id))
			weight_gradient, bias_gradient, child_id, skipdata = self.acquired_gradients_from_kids.get()
			
			### Pull from parent after consulting the policy
			if self.policy.pull_from_parent(self):
				self.application.use_parent_model(*self.pull_from_parent())

			### Merge gradients
			self.application.apply_kid_gradient(weight_gradient, bias_gradient)
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
				self.push_to_parent(*self.application.get_and_reset_acquired_gradients())

			self.merge_id += 1
		print("EXITED THE LOOP")
		self.cleanup()

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

	def cleanup(self, type="Asynchronous"):
		if type == "Asynchronous":
			if (self.parent_id != -1):
				self.send_message(self.parent_id, DISCONNECTED)

		if self.cloud_exists:
			self.ping_cloud(DISCONNECTED)

		### Stop Server Thread
		client = ServerProxy(self.own_server_address)
		client.remote_shutdown()

		### Send Done acknowledgement to Master
		self.log(self.create_log(DONE, ''))
		print("Cleanup finished")

	###-------------------------- Additional RPC functions ---------------------------------

	def push_from_child(self, weight_gradient, bias_gradient, child_id, skipdata):
		"""
		RPC function. Add the gradients obtained from child node into the queue
		"""
		# Profiling
		cur_time = time.time()
		print("Received Push request from child %s"% str(child_id))
		self.log(self.create_log(CONNECTION, 'Got gradients from child %d'%(child_id)))
		
		weight_gradient = [np.array(x)/(1.*len(self.active_children)) for x in weight_gradient]
		bias_gradient = [np.array(x)/(1.*len(self.active_children)) for x in bias_gradient]

		self.acquired_gradients_from_kids.put([weight_gradient, bias_gradient, child_id, skipdata])
		print("Time taken for push request", time.time()-cur_time)

	def pull_from_child(self, child_id):
		"""
		RPC function. Return the model (weights and biases) to the child node
		"""
		print("Received Pull request from child %s"% str(child_id))
		self.log(self.create_log(CONNECTION,'Got pull request from child %d'%(child_id)))
		model = self.application.get_model()
		model[0] = [x.tolist() for x in model[0]]
		model[1] = [x.tolist() for x in model[1]]
		return model

	def receive_message(self, sender_id, msg):
		"""
		Abstract Method Implementation
		RPC function called by child nodes. Keep track of the child's current status
		"""
		self.log(self.create_log(CONNECTION,'Received message from node id %d: %s'%(sender_id, msg)))   

		if (sender_id in self.children):
			self.child_ever_connected = True
		if ((msg == CONNECTED) and (sender_id in self.children)): 
			self.active_children.add(sender_id)
		elif ((msg == DISCONNECTED) and (sender_id in self.children)): 
			self.active_children.remove(sender_id)
			
			if len(self.active_children) == 0:
				self.cleanup()
		
		if msg == FINISH:
			# Debugging Purpose assert
			assert sender_id in self.children
			self.finish_pending_lock.acquire()
			self.finish_pending -= 1
			self.finish_pending_lock.release()

		if msg == START:
			self.start = True

		if msg == DONE:
			print("Received DONE Signal")
			for child in self.children:
				print("Sending DONE Signal to child ", child)
				self.send_message(child, DONE)
			self.done = True