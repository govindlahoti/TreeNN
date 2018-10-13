"""
Script for parsing logs
"""

import sys
import os
sys.path.append(os.path.abspath('../utility'))

import json
import argparse

from const import *

def parse_log(log_file):

	f = open(log_file, 'r')

	for row in f:
		data = json.loads(row)

		if data[TYPE] == STATISTIC and data[NODE_ID] == 2:
			print(data[PAYLOAD])

	f.close()

if __name__ == '__main__':
	
	parser = argparse.ArgumentParser()
	parser.add_argument("-f","--log_file", type=str, help="Path to the log file", required=True )
	args = parser.parse_args()

	parse_log(args.log_file)