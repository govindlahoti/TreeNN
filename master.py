"""
Code which runs on the master machine.
Logs into slave machines and triggers Nodes on the slave machines.
Starts an RPC server to receive logs about the simulation from the slaves

Run this as:
	python3 master.py -c network.yaml
"""

import sys
import argparse
import threading

from utility.util import *
from utility.const import *

from xmlrpc.client import ServerProxy
from xmlrpc.server import SimpleXMLRPCServer

server, own_address, nodes = None, None, None

def shutdown_thread():
	"""
	Shuts down RPC server
	"""
	global server
	server.shutdown()

def remote_shutdown():
	"""
	Cannot shut down the RPC server from the current thread
	Hence we need to create a separate thread to shut it down
	"""
	t = threading.Thread(target=shutdown_thread)
	t.start()

def log_report(log):
	"""
	Slave will call this function to send reports
	"""
	global nodes,own_address

	print(log, file=globals()["log_file"])
	log = json.loads(log)
	
	if log[TYPE] == DONE:
		nodes.remove(log[NODE_ID])
		if len(nodes) == 0:
			remote_shutdown()

def start_server(own_address,e):
	"""
	Start XML RPC server on the Master machine
	"""
	global server
	
	try:
		server = SimpleXMLRPCServer(own_address, allow_none=True)
		server.register_function(log_report)
		server.register_function(remote_shutdown)
		
		globals()["server_status"] = 1
		e.set()
		print('RPC server started at http://%s:%d'%own_address)
		
		server.serve_forever()
	except:
		globals()["server_status"] = -1
		print("%s:%d already in use"%own_address)
		e.set()
		exit(0)
		
if __name__ == '__main__':
	"""
	1. Fetch own ip address, port for running the RPC server is set to 9000
	2. Read yaml file and get the configuration data
	3. Start RPC server to receive logs
	"""
	
	parser = argparse.ArgumentParser()
	parser.add_argument("-n","--expname", type=str, help="Experiment name", required=True)
	parser.add_argument("-f","--config", type=str, help="Path to network config yaml file", required=True )
	parser.add_argument("-d","--docker", type=int, help="Boolean indicating to run in container mode",
								default=1, choices=[1, 0])
	parser.add_argument("-t","--trigger", type=int, help="Boolean indicating to trigger scripts (for debugging purposes)", 
								default=1, choices=[1, 0])
	parser.add_argument("-l","--log", type=int, help="Boolean indicating to generate log", 
								default=1, choices=[1,0])
	parser.add_argument("-i","--ip", type=str, help="IP address on which the Master RPC server should run",
								default=get_ip())

	args = parser.parse_args()

	own_address = (args.ip,MASTER_RPC_SERVER_PORT)

	globals()["log_file"] = open('logs/%s.log'%args.expname,'a') if args.log == 1 else sys.stdout
	globals()["server_status"] = 0
	
	e = threading.Event()
	server_thread = threading.Thread(target=start_server,args=(own_address,e,))
	server_thread.start()

	while not e.isSet():
		continue
	if globals()["server_status"] < 0: 
		exit(0)

	print(globals()["server_status"], e.wait())

	if args.trigger==1:
		try:
			data, machine_info = read_yaml(own_address,args.config,args.docker)
		except:
			print("Error in parsing configuration")
			exit(0)

		nodes = set(list(data.keys()))
		trigger_slaves(args.expname, data, machine_info, args.docker)
