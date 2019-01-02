"""
Script for parsing logs
"""

import os
import sys
sys.path.append(os.path.abspath('../utility'))

import yaml
import json
import argparse
import humanize

from const import *

def parse_logs(log_file, config):

	f = open(log_file, 'r')

	stats = {}
	link_stats = {}

	accuracy_lag = {}

	for row in f:
		data = json.loads(row)
		node_id, payload = data[NODE_ID], data[PAYLOAD]

		if data[TYPE] == STATISTIC:

			## Calculate Runtime and Process time
			if config[node_id]['is_worker']:
				if node_id in stats:
					stats[node_id][RUNTIME] += payload[RUNTIME]
					stats[node_id][PROCESS_TIME] += payload[PROCESS_TIME] 
				else:
					stats[node_id] = {
						RUNTIME 		: payload[RUNTIME],
						PROCESS_TIME	: payload[PROCESS_TIME],
					}
					
			
		elif data[TYPE] == CONNECTION:

			## Calculate network cost
			if NETWORK_COST in payload:
				parent_id = config[node_id]['parent_id']
				
				if (parent_id, node_id) in link_stats:
					link_stats[(parent_id, node_id)] += payload[NETWORK_COST]
				else:
					link_stats[(parent_id, node_id)] = payload[NETWORK_COST]


	f.close()

	for link in link_stats:
		link_stats[link] = humanize.naturalsize(link_stats[link], gnu=True)

	print(link_stats)

def read_yaml(config_file,is_docker=1):
	"""
	Read yaml from config file
	"""

	with open(config_file, 'r') as f:
		
		raw_data = yaml.load(f.read())
		data = {}
		machine_info = raw_data['machine']

		default_delay = raw_data['default_delay']

		for x in raw_data['nodes']:
			data[x['id']] = x
			data[x['id']]['is_worker'] = True
			data[x['id']]['ip'] = machine_info[x['machine']]['ip']
			data[x['id']]['own_address'] = (data[x['id']]['ip'], x['port'])

			default_fields = ['mini_batch_size','window_interval','window_limit','epochs_per_window','kafka_server','test_directory']
			if is_docker==1:
				default_fields.extend(['cpus','memory','host_test_directory','docker_image'])

			for field in default_fields:
				x[field] = raw_data['default_'+field] if field not in x else x[field]
			
		for x in raw_data['nodes']:
			
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


if __name__ == '__main__':
	
	parser = argparse.ArgumentParser()
	parser.add_argument("-f","--log_file", type=str, help="Path to the log file", required=True)
	parser.add_argument("-c","--config_file", type=str, help="Path to the config file", required=True)
	args = parser.parse_args()

	config = read_yaml(args.config_file, 1)
	parse_logs(args.log_file, config)