"""
Run this as 
python main.py setup.yaml
"""

import yaml
import sys
import netifaces as ni
import json
from time import sleep
import threading
from paramiko import client
from xmlrpc.server import SimpleXMLRPCServer

# Obtain the meta-data for all the nodes
data = {}

# Class for ssh into the machine and trigger nodes
class ssh:

	def __init__(self, address, username, password):
		print("Connecting to %s@%s..."%(username,address))
		try:
			self.client = client.SSHClient()
			self.client.set_missing_host_key_policy(client.AutoAddPolicy())
			self.client.connect(address, username=username, password=password, look_for_keys=False)
			print("Connected to %s@%s"%(username,address))
		except:
			self.client = None
			print("Authentication failed")
 	
	def trigger_node(self, node_id, node_data):
		if(self.client):
			
			### *** Edit command for triggering node here
			command = "cd Simulator/TreeNN && python3 slave.py %d '%s' &"%(node_id, node_data)
			
			print("Running command: %s"%command)
			stdin, stdout, stderr = self.client.exec_command(command)
			while not stderr.channel.exit_status_ready():
				if stderr.channel.recv_ready():
					alldata = stderr.channel.recv(1024)
					prevdata = b"1"
					while prevdata:
						print(str(alldata, "utf8"))
						prevdata = stderr.channel.recv(1024)
						alldata += prevdata
 
		else:
			print("Connection not opened")

	def disconnect(self):
		if(self.client):
			self.client.close()

# Read yaml from config file
def read_yaml(master_address):
	global data
	# Read the spec file passed as command line argument
	with open(sys.argv[1], 'r') as f:
		raw_data = yaml.load(f.read())

		# Extract the data from the yaml file

		# Obtain the default values of properties
		default_delay = raw_data['default_delay']

		for x in raw_data['nodes']:
			data[x['id']] = x
			data[x['id']]['is_worker'] = True
			data[x['id']]['own_address'] = (x['ip'], x['port'])
		 
			default_fields = ['push_interval','pull_interval','epoch_limit','batch_size']
			for field in default_fields:
				x[field] = raw_data['default_'+field] if field not in x else x[field]
			
			if 'file_name' in x:
				data[x['id']]['file_name'] = x['file_name']
			
		for x in raw_data['nodes']:
			data[x['id']]['master_address'] = 'http://%s:%d'%master_address
			
			if 'parent_id' in x:
				data[x['id']]['parent_address'] = 'http://%s:%d'%(data[x['parent_id']]['ip'], data[x['parent_id']]['port'])
				data[x['parent_id']]['is_worker'] = False				
			else:
				data[x['id']]['parent_id'] = -1
				data[x['id']]['parent_address'] = None

		 
		# Obtain the network latency information
		for x in data:
			data[x]['delays'] = {}
			data[x]['addresses'] = {}
			for y in data:
				data[x]['delays'][y] = default_delay # y is node id
				data[x]['addresses'][y] = 'http://%s:%d'%(data[y]['ip'], data[y]['port'])

		for x in raw_data['delays']:
			data[x['src_id']]['delays'][x['dest_id']] = x['delay']
		
# Log into machine and trigger node
def trigger_scripts():
	# Can do a group by to login into the machine once and trigger all the scripts
	for node_id in data:
		# Create a connection. Format: IP address, username, password
		connection = ssh(data[node_id]['ip'],'aniket','08081997ani')
		connection.trigger_node(node_id, json.dumps(data[node_id]))
		connection.disconnect()
		sleep(1)

# Start XML RPC server on the Master machine
def start_server(own_address):
	print('RPC server started at http://%s:%d'%own_address)
	server = SimpleXMLRPCServer(own_address, allow_none=True)
	server.register_function(log_report)
	server.serve_forever()

# Get logs from nodes
def log_report(log):
	print(log)

# Get own IP
def get_ip():
	ifaces = ni.interfaces()
	ifaces.sort(reverse=True)
	for iface in ifaces:
		try:
			return(ni.ifaddresses(iface)[ni.AF_INET][0]['addr'])
		except:
			continue

if __name__ == '__main__':
	
	if len(sys.argv) < 2:
		print("Format: python3 main.py <yaml file>")
		exit(0)

	own_address = (get_ip(),9000)
	read_yaml(own_address)
	server_thread = threading.Thread(target=start_server,args=(own_address,))
	server_thread.start()
	trigger_scripts()
	server_thread.join()