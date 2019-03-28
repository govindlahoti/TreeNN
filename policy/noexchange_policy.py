"""
An implementation of abstract class Policy
The model is NEVER pulled from and pushed to the parent 
"""

from policy.policy import Policy

class NoExchangePolicy(Policy):

	def __init__(self,**kwargs):
		super().__init__()

	def _pull_from_parent(self, node):
		return False

	def _push_to_parent(self, node):
		return False