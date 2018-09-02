"""
Run this as 
python master.py setup.yaml
"""

import sys
import threading

from util import *
from simulator import start_server,log_report

from xmlrpc.client import ServerProxy
from xmlrpc.server import SimpleXMLRPCServer

server, own_address, nodes = None, None, None

def shutdown_thread():
	global server
	server.shutdown()

def remote_shutdown():
	t = threading.Thread(target=shutdown_thread)
	t.start()

def log_report(log):
	global nodes,own_address

	print(log)
	log = json.loads(log)
	
	if log["type"] == "DONE":
		nodes.remove(log["node_id"])
		if len(nodes) == 0:
			remote_shutdown()

# Start XML RPC server on the Master machine
def start_server(own_address):
	global server
	print('RPC server started at http://%s:%d'%own_address)
	server = SimpleXMLRPCServer(own_address, allow_none=True)
	server.register_function(log_report)
	server.register_function(remote_shutdown)
	server.serve_forever()

if __name__ == '__main__':
	
	if len(sys.argv) < 2:
		print("Format: python3 master.py <yaml file>")
		exit(0)

	own_address = (get_ip(),9000)
	data = read_yaml(own_address,sys.argv[1])
	nodes = set(list(data.keys()))
	
	server_thread = threading.Thread(target=start_server,args=(own_address,))
	server_thread.start()
	trigger_scripts()
