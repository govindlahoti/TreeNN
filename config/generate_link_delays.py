"""
Reads tree hierarchy of nodes and generates link delays from a Gaussian
"""

import yaml
import argparse
from collections import OrderedDict

import numpy as np
np.random.seed(1)

def read_hierarchy(file_name):

	tree = []
	with open(file_name, 'r') as f:
		f.readline()
	
		for row in f:
			tree.append(tuple([int(x) for x in row.split(' ')]))
	
	return tree

def sample_link_delays(tree, mean, variance):
	delays = []

	for (src_id, dest_id) in tree:
		delays.append(
			{
				'src_id'	: src_id,
				'dest_id'	: dest_id,	
				'delay'		: float(np.random.normal(mean,variance))
			}
		)

	return {'delays': delays}


if __name__ == '__main__':
	
	parser = argparse.ArgumentParser()
	parser.add_argument("-f","--tree", type=str, help="Text file containing hierarchy")
	parser.add_argument("-m","--mean", type=str, help="Mean of ", required=True)
	parser.add_argument("-v","--variance", type=str, help="Variance of Gaussian", required=True)
	args = parser.parse_args()

	tree = read_hierarchy(args.tree)
	delays = sample_link_delays(tree, args.mean, args.variance)
	
	with open('delays.yml', 'w') as yaml_file:
		yaml.dump(delays, yaml_file, default_flow_style=False)	