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

from node import *

def main():
	
	parser = argparse.ArgumentParser()
	parser.add_argument("-ni","--node-id", type=str, dest="node_id", help="Node id", required=True)
	parser.add_argument("-nd","--node-data", type=str, dest="node_data", help="Node data", required=True)
	parser.add_argument("-k","--kafka-server", type=str, dest="kafka_server", help="Address of kafka server", default=KAFKA_SERVER_ADDRESS)
	args = parser.parse_args()

	print("Initiating node %s"%args.node_id)
	data = json.loads(args.node_data.replace('\'','\"'))

	node = None
	if data['is_worker'] is None:
		node = Cloud(data, args.kafka_server)
	else:
		if data['is_worker']:
			node = Worker(data, args.kafka_server)
		else:
			node = ParameterServer(data)
	
	threads = node.init_threads()

	for t in threads:
		t.join()
	
if __name__ == '__main__':
	main()