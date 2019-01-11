"""
An implementation of abstract class Policy
On consulting this policy, the model is pulled from and pushed to the parent after a fixed interval of time, say once every hour.
"""


import time
from policy.policy import Policy

class TimePolicy(Policy):

	def __init__(self,**kwargs):
		super().__init__()
		self.pull_interval = kwargs['pull_interval']
		self.push_interval = kwargs['push_interval']

		self.pull_timestamp = 0
		self.push_timestamp = 0

	def _pull_from_parent(self, node):
		t = time.time()
		if t - self.pull_timestamp > self.pull_interval:
			self.pull_timestamp = t
			return True
		else:
			return False

	def _push_to_parent(self, node):
		t = time.time()
		if t - self.push_timestamp > self.push_interval:
			self.push_timestamp = t
			self.pulled = False
			return True
		else:
			return False