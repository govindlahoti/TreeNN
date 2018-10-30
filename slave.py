"""
This wrapper script is placed on the slave machine and is triggered by an external master machine
It is expected that the Node files and Application files are available to the slave on the machine

Can be run individually as:
	python3 slave.py -ni <node_id> -nd <node_data>
But the above method is not advisable as the Simulation has additional dependencies of a Master machine 
RPC server (for reporting) and the parent-child hierarchy defined by the configuration file

1. Converts node data provided as arguments to a dictionary
2. Initializes threads for the node and waits for them to complete execution
"""

import json
import time
import argparse

from utility.const import KAFKA_SERVER_ADDRESS

from node.parameter_server import ParameterServer
from node.worker import Worker

def main():
	
	parser = argparse.ArgumentParser()
	parser.add_argument("-ni","--node_id", type=str, help="Node id", required=True)
	parser.add_argument("-nd","--node_data", type=str, help="Node data", required=True)
	parser.add_argument("-k","--kafka_server", type=str, help="Address of kafka server", default=KAFKA_SERVER_ADDRESS)
	args = parser.parse_args()

	print("Initiating node %s"%args.node_id)
	data = json.loads(args.node_data.replace('\'','\"'))

	node = Worker(data, args.kafka_server) if data['is_worker'] else ParameterServer(data)
	
	threads = node.init_threads()

	for t in threads:
		t.join()
	
if __name__ == '__main__':
	main()