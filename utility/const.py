"""
Contains all hardcoded configuration parameters
"""

### master.py
MASTER_RPC_SERVER_PORT = 9000

### Sensor Data rate, interval between two data points (Assumed to be a constant, can be sampled from an underlying distribution)
DATA_RATE = 1e-3
KAFKA_SERVER_ADDRESS = '192.168.43.18:9092'

### util.py
TRIGGER_NODE_COMMAND = "cd Simulator/TreeNN && python3 slave.py -ni %d -nd \"%s\" -k %s &"

TRIGGER_CONTAINER_COMMAND = "docker run -it \
-m %s --cpus=%f \
-e NODE_ID='%d' \
-e NODE_DATA=\"%s\" \
-e KAFKA_SERVER='%s' \
-p %d:%d \
--net=host \
--volume %s:%s \
--name %s_c%d \
--detach \
%s"

### constants for Logging and Reporting
CONNECTION 	= 'CONN'
STATISTIC  	= 'STAT'
DONE		= 'DONE'
PROCESSED	= 'PROC'
PULLED		= 'PULL'
PUSHED		= 'PUSH'
MERGED 		= 'MERG'

NODE_ID	= 'node_id'
TYPE 	= 'type'
PAYLOAD = 'payload'

MERGE_ID = 'Merge ID'
CHILD_ID = 'Child ID' 
SKIP_TEST_DATA = 'Skip Test data'
PRE_MERGE_ACCURACY = 'Pre Merge Accuracy'
POST_MERGE_ACCURACY = 'Post Merge Accuracy'

TIMESTAMP = 'timestamp'
WINDOW_ID = 'Window ID'
RUNTIME = 'Runtime'	
PROCESS_TIME = 'Process time'
MEMORY_USAGE = 'Memory Usage'
ACCURACY = 'Accuracy'	
DATAPOINTS = 'Datapoints'

NETWORK_COST = 'Network Cost'

### Communication protocol between Parameter server and Child node
CONNECTED 	 = 'connected'
DISCONNECTED = 'disconnected'

REDSTR = "\033[01;31m%s\033[00m"
GREENSTR = "\033[1;32m%s\033[00m"
CYANSTR = "\033[1;36m%s\033[00m"
CONTAINERSTR = CYANSTR%("\n\n################\n" + "Container: %d" + "\n################")
NODESTR = CYANSTR%("\n\n################\n" + "Node: %d" + "\n################")