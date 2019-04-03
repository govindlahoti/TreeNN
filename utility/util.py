"""
Contains utility functions for master.py
1. read_yaml(): Read yaml file and store configuration data
2. trigger_scripts(): Log into slave machines and trigger nodes
3. get_ip(): Get own IP address
"""

import yaml
import json
import threading

from time import sleep
from paramiko import client
from collections import OrderedDict

from utility.const import *

class ssh:
	"""
	Class for ssh into the slave machine and trigger nodes
	"""

	def __init__(self, address, username, password, is_thread):

		print("Connecting to %s@%s ..."%(username,address))
		try:
			self.client = client.SSHClient()
			self.client.set_missing_host_key_policy(client.AutoAddPolicy())
			self.client.connect(address, username=username, password=password, look_for_keys=False)
			self.is_thread = is_thread
			print("Connected to %s@%s"%(username,address))
		except Exception as e:
			self.client = None
			print("Authentication failed: %s"%e)
 	
	def trigger_server_node(self, node_id, node_data, kafka_server):
		command = None

		if self.is_thread:
			command = TRIGGER_THREAD_NODE_COMMAND%(node_id, node_data, kafka_server)
		else:
			command = TRIGGER_NODE_COMMAND%(node_id, node_data, kafka_server)
		self.run_command(command, node_id)

	def trigger_container(self, expname, node_id, node_data, port, kafka_server, cpus, memory, test_directory, host_test_directory, docker_image):
		command = None

		if self.is_thread:
			command = TRIGGER_THREAD_CONTAINER_COMMAND%(memory, cpus, node_id, node_data, kafka_server, port, port, host_test_directory, test_directory, expname, node_id, docker_image)	
		else:
			command = TRIGGER_CONTAINER_COMMAND%(memory, cpus, node_id, node_data, kafka_server, port, port, host_test_directory, test_directory, expname, node_id, docker_image)
		self.run_command(command, node_id)
		
	def run_command(self, command, node_id=None):

		if(self.client):	
			print("Running command: %s"%command)
			stdin, stdout, stderr = self.client.exec_command(command, get_pty=True)
			while not stdout.channel.exit_status_ready():
				if stdout.channel.recv_ready():
					alldata = stdout.channel.recv(1024)
					prevdata = b"1"
					while prevdata:
						print(node_id, prevdata.decode("utf-8"))
						prevdata = stdout.channel.recv(1024)
						alldata += prevdata
		else:
			print("Connection not opened")

	def disconnect(self):

		if(self.client):
			self.client.close()

def read_yaml(master_address,config_file,is_docker, include_cloud):
	"""
	Read yaml from config file
	master_address is sent to slaves so that they can report back the logs
	"""

	with open(config_file, 'r') as f:
		
		raw_data = yaml.load(f.read())
		data = OrderedDict()
		machine_info = raw_data['machine']
		application_arguments = raw_data['application_arguments']
		policy = raw_data['policy']

		default_bandwidth = raw_data['default_bandwidth']

		sensors = []

		for x in raw_data['nodes']:
			data[x['id']] = x
			data[x['id']]['is_worker'] = True
			data[x['id']]['ip'] = machine_info[x['machine']]['ip']
			data[x['id']]['own_address'] = (data[x['id']]['ip'], x['port'])

			default_fields = ['window_interval','window_limit','kafka_server','test_directory','policy','args']
			if is_docker==1:
				default_fields.extend(['cpus','memory','host_test_directory','docker_image'])

			for field in default_fields:
				data[x['id']][field] = raw_data['default_'+field] if field not in x else x[field]
			
			data[x['id']]['policy'] = policy[data[x['id']]['policy']]
			if data[x['id']]['policy']['args'] is None:
				data[x['id']]['policy']['args'] = {} 
			data[x['id']]['cloud_exists'] = False 

		for x in raw_data['nodes']:
			data[x['id']]['master_address'] = 'http://%s:%d'%master_address
			
			data[x['id']]['application_arguments'] = application_arguments[data[x['id']]['args']]
			if 'parent_id' in x:
				data[x['id']]['parent_address'] = 'http://%s:%d'%(data[x['parent_id']]['ip'], data[x['parent_id']]['port'])
				data[x['parent_id']]['is_worker'] = False				
			else:
				data[x['id']]['parent_id'] = -1
				data[x['id']]['parent_address'] = None

		### Obtain the network bandwidth information
		for x in data:
			data[x]['bandwidths'] = {}
			data[x]['addresses'] = {}
			for y in data:
				data[x]['bandwidths'][y] = default_bandwidth 
				data[x]['addresses'][y] = 'http://%s:%d'%(data[y]['ip'], data[y]['port'])

		for x in raw_data['bandwidths']:
			data[x['src_id']]['bandwidths'][x['dest_id']] = x['bandwidth']
			data[x['dest_id']]['bandwidths'][x['src_id']] = x['bandwidth']

		## Run a simulation of the cloud
		if include_cloud == 1:
			data[-1] = raw_data['cloud']
			data[-1]['id'] = -1
			data[-1]['ip'] = machine_info[raw_data['cloud']['machine']]['ip']
			data[-1]['own_address'] = (data[-1]['ip'], raw_data['cloud']['port'])
			data[-1]['is_worker'] = None

			default_fields = ['window_interval','window_limit','kafka_server','test_directory','policy','args']
			if is_docker==1:
				default_fields.extend(['cpus','memory','host_test_directory','docker_image'])

			for field in default_fields:
				data[-1][field] = raw_data['default_'+field] if field not in data[-1] else data[-1][field]

			data[-1]['policy'] = NOEXCHANGE_POLICY
			data[-1]['master_address'] = 'http://%s:%d'%master_address
			data[-1]['application_arguments'] = application_arguments[data[-1]['args']]

			data[-1]['parent_id'] = -1
			data[-1]['parent_address'] = None

			data[-1]['addresses'] = {}				
			data[-1]['bandwidths'] = {}

			for x in data:
				data[x]['cloud_exists'] = True
				data[x]['addresses'][-1] = 'http://%s:%d'%(data[-1]['ip'], data[-1]['port'])
				data[x]['bandwidths'][-1] = 10e10
				if data[x]['is_worker']:
                                        sensors.extend(data[x]['sensors'])
			data[-1]['cloud_exists'] = False
			data[-1]['sensors'] = sensors
			
	return data,machine_info

def _trigger_docker(expname, data, machine, node_id, is_thread=False):
	connection = ssh(machine['ip'],machine['username'],machine['password'], is_thread)
	connection.trigger_container(expname,
								 node_id, 
								 json.dumps(data).replace('\"','\''), 
								 data['port'],
								 data['kafka_server'],
								 data['cpus'],
								 data['memory'],
								 data['test_directory'],
								 data['host_test_directory'],
								 data['docker_image'])
	return connection

def _trigger_server(data, machine, node_id, is_thread=False):
	connection = ssh(machine['ip'],machine['username'],machine['password'], is_thread)
	connection.trigger_server_node(node_id, json.dumps(data).replace('\"','\''), data['kafka_server'])
	return connection

def trigger_slaves(expname, data, machine_info, use_docker):
	"""
	Log into machine and trigger node
	"""

	# Can do a group by to login into the machine once and trigger all the scripts
	for node_id in data:
		# Create a connection. Format: IP address, username, password
		machine = machine_info[data[node_id]['machine']]
		if use_docker == 1:
			print(CONTAINERSTR%node_id)
			connection = _trigger_docker(expname, data[node_id], machine, node_id)
			
		else:
			print(NODESTR%node_id)
			connection = _trigger_server(data[node_id], machine, node_id)
		connection.disconnect()
		sleep(1)

def trigger_threaded_slaves(expname, data, machine_info, use_docker):
	"""
	Log into machine and trigger node (handled by a separate thread)
	"""

	# Can do a group by to login into the machine once and trigger all the scripts
	for node_id in data:
		# Create a connection. Format: IP address, username, password
		machine = machine_info[data[node_id]['machine']]
		if use_docker == 1:
			print(CONTAINERSTR%node_id)
			t = threading.Thread(target=_trigger_docker,args=(expname, data[node_id], machine, node_id,True))
			t.start()			
		else:
			print(NODESTR%node_id)
			t = threading.Thread(target=_trigger_server,args=(data[node_id], machine, node_id, True))
			t.start()
		sleep(1)

def get_ip():
	"""
	Get own IP address
	"""
	import netifaces as ni

	ifaces = ni.interfaces()
	ifaces.sort(reverse=True)
	for iface in ifaces:
		try:
			return(ni.ifaddresses(iface)[ni.AF_INET][0]['addr'])
		except:
			continue

def create_log_directory(config_file, expname):
	"""
	Create logging directory and copy config yaml 
	"""
	import os
	import shutil
	import datetime

	now = datetime.datetime.now()
	dir_path = "logs/%s-%s"%(now.strftime("%Y%m%d-%H%M"), expname)

	if not os.path.exists(dir_path):
		os.makedirs(dir_path)
	shutil.copy(config_file, dir_path)

	return dir_path
