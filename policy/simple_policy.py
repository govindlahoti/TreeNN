"""
An implementation of abstract class Policy
The model is pulled from and pushed to the parent EVERY TIME the node consults this policy
"""

from policy.policy import Policy

class SimplePolicy(Policy):

	def __init__(self,**kwargs):
		super().__init__()

	def _pull_from_parent(self, node):
		return True

	def _push_to_parent(self, node):
		return True