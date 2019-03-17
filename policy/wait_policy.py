"""
An implementation of abstract class Policy
On consulting this policy, the model is pulled from and pushed to the parent after waiting for a fixed interval of time
"""


import time
from policy.policy import Policy

class WaitPolicy(Policy):

	def __init__(self,**kwargs):
		super().__init__()
		self.wait_interval = kwargs['wait_interval']
		self.start_timestamp = 0

	def _pull_from_parent(self, node):
		t = time.time()
		if t - self.start_timestamp > self.wait_interval:
			return True
		else:
			return False

	def _push_to_parent(self, node):
		t = time.time()
		if t - self.start_timestamp > self.wait_interval:
			self.pulled = False
			return True
		else:
			return False