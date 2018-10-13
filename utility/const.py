"""
Contains all hardcoded configuration parameters
"""

### master.py
MASTER_RPC_SERVER_PORT = 9000

### Sensor Data rate, interval between two data points (Assumed to be a constant, can be sampled from an underlying distribution)
DATA_RATE = 1e-3
KAFKA_SERVER_ADDRESS = 'localhost:9092'

### util.py
TRIGGER_NODE_COMMAND = "cd Simulator/TreeNN && python3 slave.py -ni %d -nd '%s' &"

### constants for Logging and Reporting
CONNECTION 	= 'CONN'
STATISTIC  	= 'STAT'
DONE		= 'DONE'

NODE_ID	= 'node_id'
TYPE 	= 'type'
PAYLOAD = 'payload'

MERGE_ID= 'Merge ID'
PRE_MERGE_ACCURACY = 'Pre Merge Accuracy'
POST_MERGE_ACCURACY = 'Post Merge Accuracy'

WINDOW_ID = 'Window ID'
RUNTIME = 'Runtime'	
PROCESS_TIME = 'Process time'
MEMORY_USAGE = 'Memory Usage'
ACCURACY = 'Accuracy'	

### Communication protocol between Parameter server and Child node
CONNECTED 	 = 'connected'
DISCONNECTED = 'disconnected'
