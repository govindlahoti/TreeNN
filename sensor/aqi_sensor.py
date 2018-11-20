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

def send_data(topic, source, data_rate, kafka_server):

	try:
		producer = KafkaProducer(bootstrap_servers=kafka_server, api_version=(0,10))	
		print("Sensor %s: Connected to Kafka"%topic)
	except NoBrokersAvailable:
		print("No Brokers are Available. Please start the Kafka server")
		exit(0)

	with open(source, 'r') as f:
		counter = 0
		for data_point in f:
			producer.send(topic, data_point.encode('utf-8'))
			counter+=1
			# if counter%100 == 0: print(counter)
			sleep(data_rate)
			
if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("-sid","--sensor-id", type=str, dest="sensor_id", help="Sensor ID and also the topic on which the sensor will post data", required=True)
	parser.add_argument("-s","--source", type=str, help="Source file using which sensor dumps data", required=True)
	parser.add_argument("-d","--data-rate", type=float, default=DATA_RATE, dest="data_rate", help="Time interval between sending two data measurements")
	parser.add_argument("-k","--kafka-server", type=str, default=KAFKA_SERVER_ADDRESS, dest="kafka_server", help="Kafka Server address")
	args = parser.parse_args()
	
	send_data(args.sensor_id, args.source, args.data_rate, args.kafka_server)