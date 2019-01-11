"""
An implementation of abstract class Policy
On consulting this policy, the model is pulled from and pushed to the parent after processing a fixed number of windows
To be used ONLY by worker nodes
"""

from policy.policy import Policy

class WindowPolicy(Policy):

	def __init__(self,**kwargs):
		super().__init__()
		self.pull_count = kwargs['pull_count']
		self.push_count = kwargs['push_count']

	def _pull_from_parent(self, node):
		if node.window_count % self.pull_count == 0:
			return True
		else:
			return False

	def _push_to_parent(self, node):
		if node.window_count % self.push_count == 0:
			return True
		else:
			return False