from SimpleXMLRPCServer import SimpleXMLRPCServer
import time
from queue import Queue
import xmlrpclib
import thread
from network import *

np.random.seed(42)
data_trend = np.random.normal(0, 1, (48, 729))

def get_data():
	def get_y(x):
		return sigmoid(np.dot(data_trend, x))
		
	x_vals = [np.random.normal(0, 1, (729, 1)) for _ in range(20)]
	y_vals = map(get_y, x_vals)

	return zip(x_vals, y_vals)

data = get_data()

class Node:

	def __init__(self, id, parent_id, own_address, parent_address, is_worker, worker_pull_interval, worker_push_interval):
		self.id = id
		self.own_address = own_address
		self.parent_id = parent_id
		self.parent_address = parent_address
		self.is_worker = is_worker
		self.connected_with_parent = False
		self.e = 0
		self.prev_e = 0
		self.e_lock = threading.Lock()
		self.worker_push_interval = worker_push_interval
		self.worker_pull_interval = worker_pull_interval
		self.network = Network([729, 48])

		self.log_file = open(str(self.id) + '.log', 'a')
		
		self.acquired_gradients_from_kids = Queue()

		thread.start_new_thread(self.comsume_gradients_from_kids_thread, ())
		thread.start_new_thread(self.run_rpc_server_thread, ())
		thread.start_new_thread(self.run_sharing_logic_thread, ())


	def get_parent(self):
		if self.connected_with_parent:
			return self.parent

		elif not self.parent_address:
			return None
		
		else:
			self.parent = xmlrpclib.ServerProxy(self.parent_address, allow_none=True)
			
			while True:
				self.log('Node {} : Waiting for parent {} ... '.format(self.id, self.parent_address))
				try:
					self.parent.pull_from_child(self.id)
					self.log('Node {} : Connected with parent {}'.format(self.id, self.parent_address))
					self.connected_with_parent = True
					break
				except Exception, e:
					print e
					time.sleep(1)

			return self.parent


	def push_from_child(self, weight_gradient, bias_gradient, child_id):
		self.log('Got gradients from child ' + str(child_id))
		weight_gradient = [np.array(x) for x in weight_gradient]
		bias_gradient = [np.array(x) for x in bias_gradient]
		self.acquired_gradients_from_kids.put([weight_gradient, bias_gradient])
	

	def pull_from_child(self, child_id):
		self.log('Got pull request from child ' + str(child_id))
		model = self.network.get_model()
		model[0] = [x.tolist() for x in model[0]]
		model[1] = [x.tolist() for x in model[1]]
		return model
	

	def push_to_parent(self, weight_gradient, bias_gradient):
		if not self.parent_address:
			return
		weight_gradient = [x.tolist() for x in weight_gradient]
		bias_gradient = [x.tolist() for x in bias_gradient]
		self.get_parent().push_from_child(weight_gradient, bias_gradient, self.id)


	def pull_from_parent(self):
		if not self.parent_address:
			return
		model = self.get_parent().pull_from_child(self.id)
		model[0] = [np.array(x) for x in model[0]]
		model[1] = [np.array(x) for x in model[1]]
		self.log('Got model from parent ' + str(self.parent_id))
		return model

	def get_loss(self):
		return float(self.network.evaluate(data))

	def comsume_gradients_from_kids_thread(self):
		while True:
			weight_gradient, bias_gradient = self.acquired_gradients_from_kids.get()
			self.network.apply_kid_gradient(weight_gradient, bias_gradient)
			self.e_lock.acquire()
			self.e += 1
			self.e_lock.release()


	def run_rpc_server_thread(self):
		server = SimpleXMLRPCServer(self.own_address, allow_none=True)
		server.register_function(self.push_from_child, "push_from_child")
		server.register_function(self.pull_from_child, "pull_from_child")
		server.register_function(self.get_loss, "get_loss")
		server.serve_forever()


	def run_sharing_logic_thread(self):
		# data = self.get_data()
		
		while True:
			if self.is_worker:

				if self.e % self.worker_pull_interval == 0:
					if self.parent_address:
						self.network.use_parent_model(*self.pull_from_parent())

				self.network.SGD(data, test_data=data, epochs=1)

				if self.e % self.worker_push_interval == 0:
					if self.parent_address:
						self.push_to_parent(*self.network.get_and_reset_acquired_gradients())

				self.e += 1
		
			else:
				self.e_lock.acquire()
				e = self.e
				self.e_lock.release()
				
				if e != self.prev_e:
					self.prev_e = e
				else:
					time.sleep(1)
					continue
				
				if e % self.worker_pull_interval == 0:
					if self.parent_address:
						self.network.use_parent_model(*self.pull_from_parent())

				if e % self.worker_push_interval == 0:
					if self.parent_address:
						self.push_to_parent(*self.network.get_and_reset_acquired_gradients())

			self.log('Epoch {} Loss = {}'.format(self.e, self.network.evaluate(data)))


	def log(self, s):
		self.log_file.write(s +'\n')
		self.log_file.flush()
