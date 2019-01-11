"""
An abstract class for implementing a model-sharing policy
"""

from abc import ABC, abstractmethod

class Policy(ABC):

	def __init__(self):
		"""
		Every policy maintains the number of updates done to the model
		Updates do not include pushing a model to the parent
		For Parameter Servers: Updates include pulls from the parent and merges completed
		For Worker nodes: Updates include only the number of pulls from the parent
		Updates restores to 0 on pushing the model to the parent

		To avoid multiple pulls without a push (thus leading to loss of information), a boolean
		is maintained.
		"""
		self.updates = 0
		self.pulled = False

	@abstractmethod
	def _pull_from_parent(self, node):
		print("Method not implemented")

	@abstractmethod
	def _push_to_parent(self, node):
		print("Method not implemented")

	def pull_from_parent(self, node):
		if node.parent_address and not self.pulled:
			self.pulled = self._pull_from_parent(node)

			if self.pulled:
				self.updates += 1

			return self.pulled
		else:
			return False

	def push_to_parent(self, node):
		if node.parent_address:
			pushed = self._push_to_parent(node)
			
			if pushed:
				self.pulled = False
				self.updates = 0

			return pushed
		else:
			return False