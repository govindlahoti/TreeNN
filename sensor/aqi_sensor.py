"""
This script emulates a sensor which acts as a KafkaProducer
Currently it is reading a file and dumping data into Kafka
"""

import sys
import os
sys.path.append(os.path.abspath('../utility'))

import argparse
from time import sleep

from const import DATA_RATE, KAFKA_SERVER_ADDRESS

from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable

def send_data(topic, source, data_rate):

	try:
		producer = KafkaProducer(bootstrap_servers=KAFKA_SERVER_ADDRESS)	
	except NoBrokersAvailable:
		print("No Brokers are Available. Please start the Kafka server")
		exit(0)

	with open(source, 'r') as f:
		for data_point in f:
			producer.send(topic, data_point.encode('utf-8'))
			sleep(data_rate)

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("-sid","--sensor_id", type=str, help="Sensor ID")
	parser.add_argument("-t","--topic", type=str, help="Topic on which the sensor will post, basically the node id of the worker", required=True)
	parser.add_argument("-s","--source", type=str, help="Source file using which sensor dumps data", required=True)
	parser.add_argument("-d","--data_rate", type=float, default=DATA_RATE, help="Time interval between sending two data measurements")
	args = parser.parse_args()
	
	send_data(args.topic, args.source, args.data_rate)