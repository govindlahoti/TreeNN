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
from collections import OrderedDict

def parse_logs(log_file, config):

	f = open(log_file, 'r')

	stats = {}
	link_stats = {}

	accuracy_lag = { node_id:OrderedDict() if not config[node_id]['is_worker'] else None for node_id in config }
	window_lags_pointer = { node_id:0 for node_id in config }

	for row in f:
		data = json.loads(row)
		node_id, payload, timestamp = data[NODE_ID], data[PAYLOAD], data[TIMESTAMP]
		parent_id = config[node_id]['parent_id']

		if data[TYPE] == STATISTIC:

			## Calculate Runtime, Process time, total datapoints
			if config[node_id]['is_worker'] and isinstance(payload,dict):
				if node_id in stats: 
					stats[node_id][RUNTIME] += payload[RUNTIME]
					stats[node_id][PROCESS_TIME] += payload[PROCESS_TIME] 
					stats[node_id][DATAPOINTS] += payload[DATAPOINTS]
				else:
					stats[node_id] = {
						RUNTIME 		: payload[RUNTIME],
						PROCESS_TIME	: payload[PROCESS_TIME],
						DATAPOINTS		: payload[DATAPOINTS]
					}
					
			## Calculate accuracy lag - Part 1 (Store window processing timestamp)
			if config[node_id]['is_worker'] and not isinstance(payload,dict):
				## Log which indicates window is processed
				window_id = int(payload.split()[1])

				if window_id not in accuracy_lag[parent_id]:
					accuracy_lag[parent_id][window_id] = { 
						node_id: timestamp 
					}
				else:
					accuracy_lag[parent_id][window_id].update({node_id:timestamp})				

		elif data[TYPE] == CONNECTION:

			## Calculate network cost
			if NETWORK_COST in payload:
				
				if (parent_id, node_id) in link_stats:
					link_stats[(parent_id, node_id)] += payload[NETWORK_COST]
				else:
					link_stats[(parent_id, node_id)] = payload[NETWORK_COST]

			## Calculate accuracy lag - Part 2 (Receive model from parent)
			if config[node_id]['is_worker'] and "Got model from parent" in payload:
				## Log which indicates model receival
				lag_pointer = window_lags_pointer[node_id]
				if lag_pointer in accuracy_lag[parent_id]:
					while lag_pointer in accuracy_lag[parent_id] and len(accuracy_lag[parent_id][lag_pointer]) == config[parent_id]['children_count']:
						accuracy_lag[parent_id][lag_pointer][node_id] = timestamp - accuracy_lag[parent_id][lag_pointer][node_id]
						lag_pointer += 1
					window_lags_pointer[node_id] = lag_pointer	


	f.close()

	for link in link_stats:
		link_stats[link] = humanize.naturalsize(link_stats[link], gnu=True)

	print(accuracy_lag)
	# print(stats)
	# print(link_stats)

def read_learning_tree(config_file):
	"""
	Read yaml from config file
	"""

	with open(config_file, 'r') as f:
		
		raw_data = yaml.load(f.read())
		tree = {}

		for x in raw_data['nodes']:
			tree[x['id']] = x
			tree[x['id']]['is_worker'] = True
			tree[x['id']]['children_count'] = 0
			
		for x in raw_data['nodes']:
			
			if 'parent_id' in x:
				tree[x['parent_id']]['is_worker'] = False
				tree[x['parent_id']]['children_count'] += 1				
			else:
				tree[x['id']]['parent_id'] = -1

	return tree


if __name__ == '__main__':
	
	parser = argparse.ArgumentParser()
	parser.add_argument("-f","--log_file", type=str, help="Path to the log file", required=True)
	parser.add_argument("-c","--config_file", type=str, help="Path to the config file", required=True)
	args = parser.parse_args()

	config = read_learning_tree(args.config_file)
	parse_logs(args.log_file, config)