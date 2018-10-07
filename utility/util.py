"""
Contains utility functions for master.py
1. read_yaml(): Read yaml file and store configuration data
2. trigger_scripts(): Log into slave machines and trigger nodes
3. get_ip(): Get own IP address
"""

import yaml
import json
from time import sleep
import netifaces as ni
from paramiko import client
from collections import OrderedDict

from utility.const import *

data = OrderedDict()

class ssh:
	"""
	Class for ssh into the slave machine and trigger nodes
	"""

	def __init__(self, address, username, password):

		print("Connecting to %s@%s ..."%(username,address))
		try:
			self.client = client.SSHClient()
			self.client.set_missing_host_key_policy(client.AutoAddPolicy())
			self.client.connect(address, username=username, password=password, look_for_keys=False)
			print("Connected to %s@%s"%(username,address))
		except Exception as e:
			self.client = None
			print("Authentication failed: %s"%e)
 	
	def trigger_node(self, node_id, node_data):

		if(self.client):
			
			command = TRIGGER_NODE_COMMAND%(node_id, node_data)

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

def read_yaml(master_address,config_file):
	"""
	Read yaml from config file
	master_address is sent to slaves so that they can report back the logs
	"""
	global data

	with open(config_file, 'r') as f:
		
		raw_data = yaml.load(f.read())

		# Obtain the default values of properties
		default_delay = raw_data['default_delay']

		for x in raw_data['nodes']:
			data[x['id']] = x
			data[x['id']]['is_worker'] = True
			data[x['id']]['own_address'] = (x['ip'], x['port'])

			default_fields = ['epoch_limit','mini_batch_size','window_size']
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

	return data
	
def trigger_scripts():
	"""
	Log into machine and trigger node
	"""

	# Can do a group by to login into the machine once and trigger all the scripts
	for node_id in data:
		# Create a connection. Format: IP address, username, password
		connection = ssh(data[node_id]['ip'],data[node_id]['username'],data[node_id]['password'])
		connection.trigger_node(node_id, json.dumps(data[node_id]))
		connection.disconnect()
		sleep(1)

def get_ip():
	"""
	Get own IP address
	"""

	ifaces = ni.interfaces()
	ifaces.sort(reverse=True)
	for iface in ifaces:
		try:
			return(ni.ifaddresses(iface)[ni.AF_INET][0]['addr'])
		except:
			continue
