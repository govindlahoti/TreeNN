from policy.policy import Policy

class AccuracyPolicy(Policy):

	def __init__(self,**kwargs):
		super().__init__()
		self.threshold = kwargs['threshold']

		self.parent_update_count = -1
		self.max_accuracy = -1

	def _pull_from_parent(self, node):
		
		current_parent_update_count = node.get_parent_update_count()

		if self.parent_update_count != current_parent_update_count and current_parent_update_count > 0:
			self.parent_update_count = current_parent_update_count
			return True
		else:
			return False

	def _push_to_parent(self, node):
		
		if self.max_accuracy == -1:
			self.max_accuracy = max(node.accuracies.values())
			return False

		if max(node.accuracies.values()) - self.max_accuracy > self.threshold:
			self.max_accuracy = -1
			return True
		else:
			return False