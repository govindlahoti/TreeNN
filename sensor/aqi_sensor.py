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

def send_data(topic):

	try:
		producer = KafkaProducer(bootstrap_servers=KAFKA_SERVER_ADDRESS)	
	except NoBrokersAvailable:
		print("No Brokers are Available. Please start the Kafka server")
		exit(0)

	### Need to supplied as an argument
	with open('../data/Cluster8_Data.csv', 'r') as f:
		for data_point in f:
			producer.send(topic, data_point.encode('utf-8'))
			sleep(DATA_RATE)

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("-sid","--sensor_id", type=str, help="Sensor ID")
	parser.add_argument("-t","--topic", type=str, help="Topic on which the sensor will post, basically the node id of the worker", required=True)
	args = parser.parse_args()
	
	send_data(args.topic)