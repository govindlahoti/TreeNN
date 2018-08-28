"""
Run this as 
python master.py setup.yaml
"""

import sys
import threading

from util import *
from simulator import start_server,log_report

# Obtain the meta-data for all the nodes

if __name__ == '__main__':
	
	if len(sys.argv) < 2:
		print("Format: python3 master.py <yaml file>")
		exit(0)

	own_address = (get_ip(),9000)
	read_yaml(own_address,sys.argv[1])
	server_thread = threading.Thread(target=start_server,args=(own_address,))
	server_thread.start()
	trigger_scripts()
	server_thread.join()