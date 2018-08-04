"""
Run this as 
python main.py setup.yaml
"""

import yaml
import sys

import json
from paramiko import client

# Obtain the meta-data for all the nodes
data = {}

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
			command = "cd Simulator/TreeNN && python slave.py %d '%s' &"%(node_id, node_data)
			print("Running command: %s"%command)
			stdin, stdout, stderr = self.client.exec_command(command)
			while not stderr.channel.exit_status_ready(): # Print data when available
				if stderr.channel.recv_ready():
					# print("here")
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

def read_yaml():
	global data
	# Read the spec file passed as command line argument
	with open(sys.argv[1], 'r') as f:
		raw_data = yaml.load(f.read())

		# extract the data from the yaml file

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
			if 'parent_id' in x:
				
				data[x['id']]['parent_address'] = 'http://{}:{}'.format(data[x['parent_id']]['ip'], data[x['parent_id']]['port'])
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
				data[x]['addresses'][y] = 'http://{}:{}'.format(data[y]['ip'], data[y]['port'])

		for x in raw_data['delays']:
			data[x['src_id']]['delays'][x['dest_id']] = x['delay']
		

def trigger_scripts():
	#Can do a group by to login into the machine once and trigger all the scripts
	for node_id in data:
		connection = ssh(data[node_id]['ip'],'aniket','08081997ani')
		connection.trigger_node(node_id, json.dumps(data[node_id]))
		connection.disconnect()


if __name__ == '__main__':
	read_yaml()
	trigger_scripts()