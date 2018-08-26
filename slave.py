import sys
import json
from parameter_server import ParameterServer
from worker import Worker
import time

def main():
	if(len(sys.argv)<3):
		print("Usage: python slave.py <node_id> <node_data>")
		exit()

	# # Spawn all the nodes as mentioned in the spec file
	print("Initiating node %s"%sys.argv[1])
	data = json.loads(sys.argv[2])

	node = Worker(data) if data['is_worker'] else ParameterServer(data)
	
	threads = node.init_threads()

	for t in threads:
		t.join()
	
	# # Run this thread indefinitely
	# while True:
	# 	# Collect reports periodically
	# 	time.sleep(100)

	#send finish signal to master

if __name__ == '__main__':
	main()