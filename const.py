"""
Contains all hardcoded configuration parameters
"""

### master.py
MASTER_RPC_SERVER_PORT = 9000

### util.py
TRIGGER_NODE_COMMAND = "cd Simulator/TreeNN && python3 slave.py %d '%s' &"

### constants for Logging and Reporting
CONNECTION 	= 'CONN'
STATISTIC  	= 'STAT'
DONE		= 'DONE'

NODE_ID	= 'node_id'
TYPE 	= 'type'
PAYLOAD = 'payload'

### Communication protocol between Parameter server and Child node
CONNECTED 	 = 'connected'
DISCONNECTED = 'disconnected'