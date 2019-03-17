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

import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from scipy.interpolate import spline

from const import *
from collections import OrderedDict

stats = {}
link_stats = {}
accuracy_lag = None

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

def parse_logs(log_file, config):

	f = open(log_file, 'r')

	accuracy_lag = { node_id:OrderedDict() if not config[node_id]['is_worker'] else None for node_id in config }
	window_lags_pointer = { node_id:0 for node_id in config }
	base_timestamp = None
	
	for row in f:

		data = json.loads(row)
		node_id, payload, timestamp = data[NODE_ID], data[PAYLOAD], data[TIMESTAMP]
		parent_id = config[node_id]['parent_id']

		if base_timestamp is None: base_timestamp = timestamp

		if data[TYPE] == STATISTIC:

			## Calculate Accuracy and timestamp
			if node_id in stats: 
				stats[node_id][TIMESTAMP].append(timestamp - base_timestamp)
				try:
					for test_file in payload[ACCURACY]:
						stats[node_id][ACCURACY][test_file].append(payload[ACCURACY][test_file])
				except:
					for test_file in payload[POST_MERGE_ACCURACY]:
						stats[node_id][ACCURACY][test_file].append(payload[POST_MERGE_ACCURACY][test_file])
			else:
				try:
					stats[node_id] = {
						ACCURACY 		: { test_file:[a] for test_file,a in payload[ACCURACY].items() },
						TIMESTAMP 		: [timestamp - base_timestamp]
					}
				except:
					stats[node_id] = {
						ACCURACY 		: { test_file:[a] for test_file,a in payload[POST_MERGE_ACCURACY].items() },
						TIMESTAMP 		: [timestamp - base_timestamp]
					}

			##Calculate Runtime, Process time and total datapoints (if node is a worker)
			if config[node_id]['is_worker']:
				if RUNTIME in stats[node_id]:
					stats[node_id][RUNTIME] += payload[RUNTIME]
					stats[node_id][PROCESS_TIME] += payload[PROCESS_TIME] 
					stats[node_id][DATAPOINTS] += payload[DATAPOINTS]
				else:
					stats[node_id][RUNTIME] = payload[RUNTIME]
					stats[node_id][PROCESS_TIME] =  payload[PROCESS_TIME]
					stats[node_id][DATAPOINTS] = payload[DATAPOINTS]

			
		elif data[TYPE] == CONNECTION:
			## Calculate network cost
			if NETWORK_COST in payload:
				if (parent_id, node_id) in link_stats:
					link_stats[(parent_id, node_id)] += payload[NETWORK_COST]
				else:
					link_stats[(parent_id, node_id)] = payload[NETWORK_COST]

		elif data[TYPE] == PROCESSED:		
			## Calculate accuracy lag - Part 1 (Store window processing timestamp)
			if config[node_id]['is_worker']:
				## Log which indicates window is processed
				window_id = payload[WINDOW_ID]

				if window_id not in accuracy_lag[parent_id]:
					accuracy_lag[parent_id][window_id] = { 
						node_id: timestamp 
					}
				else:
					accuracy_lag[parent_id][window_id].update({node_id:timestamp})				


		elif data[TYPE] == PULLED:
			## Calculate accuracy lag - Part 2 (Receive model from parent)
			if config[node_id]['is_worker']:
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

def plot_graphs(graphs_directory):

	sb.set_style("darkgrid")
	sb.set_palette(sb.color_palette(n_colors = 10))
	sb.despine()

	window_size = 10

	for node_id in stats:
		stat = stats[node_id]
		for f in stat[ACCURACY]:
			l, x = len(stat[ACCURACY][f]), stat[ACCURACY][f]
			rmse_smooth = [0 for i in range(l)]
			rmse_smooth[0:window_size] = x[0:window_size]
			rmse_smooth[l-window_size:] = x[l-window_size:]
			for i in range(window_size, l-window_size-1):
				import statistics as s
				rmse_smooth[i] = s.mean(x[i-window_size:i+window_size])
				# print(rmse_smooth)
			plt.plot(stat[TIMESTAMP], x, label=f)
		plt.title('Accuracy plot for Node %d'%node_id)
		plt.xlabel('Seconds')
		plt.ylabel('Accuracy')
		plt.legend(ncol=2, loc='lower right')
		plt.savefig('%s/accuracy-%d.jpg'%(graphs_directory,node_id))
		plt.clf()

if __name__ == '__main__':
	
	parser = argparse.ArgumentParser()
	parser.add_argument("-p","--log_path", type=str, help="Path to log directory", required=True)
	args = parser.parse_args()

	log_file, config_file = None, None
	try:
		for file in os.listdir(args.log_path):
			if file.endswith(".log"): log_file = os.path.join(args.log_path, file)
			if file.endswith(".yaml"): config_file = os.path.join(args.log_path, file)
	except FileNotFoundError:
		print("Log Directory not found"); exit(0)

	if log_file is None or config_file is None:
		print("Invalid Log directory");	exit(0)

	config = read_learning_tree(config_file)
	parse_logs(log_file, config)

	graphs_directory = os.path.join(args.log_path, "graphs")
	if not os.path.exists(graphs_directory):
		os.makedirs(graphs_directory)

	plot_graphs(graphs_directory)
	