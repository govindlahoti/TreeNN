"""
This wrapper script is placed on the slave machine and is triggered by an external master machine
It is expected that the Node files and Application files are available to the slave on the machine

Can be run individually as:
	python3 slave.py <node_id> <node_data>
But the above method is not advisable as the Simulation has additional dependencies of a Master machine 
RPC server (for reporting) and the parent-child hierarchy defined by the configuration file

1. Converts node data provided as arguments to a dictionary
2. Initializes threads for the node and waits for them to complete execution
"""

import sys
import json
from parameter_server import ParameterServer
from worker import Worker
import time

def main():
	
	if(len(sys.argv)<3):
		print("Usage: python3 slave.py <node_id> <node_data>")
		exit()

	print("Initiating node %s"%sys.argv[1])
	data = json.loads(sys.argv[2])

	node = Worker(data) if data['is_worker'] else ParameterServer(data)
	
	threads = node.init_threads()

	for t in threads:
		t.join()
	
if __name__ == '__main__':
	main()