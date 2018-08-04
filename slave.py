import sys
import json
from node import Node
import time

def main():
	if(len(sys.argv)<3):
		print("Usage: python slave.py <node_id> <node_data>")
		exit()

	# # Spawn all the nodes as mentioned in the spec file
	print("Initiating node %s"%sys.argv[1])
	data = json.loads(sys.argv[2])
	node = Node(data)
	threads = node.init_threads()

	for t in threads:
		t.join()
		
	# # Run this thread indefinitely
	# while True:
	# 	# Collect reports periodically
	# 	time.sleep(100)

if __name__ == '__main__':
	main()