import argparse
from time import sleep

from kafka import KafkaProducer

def send_data(topic):
	producer = KafkaProducer(bootstrap_servers='localhost:9092')	
	data_rate = 1e-1
	with open('../data/Cluster8_Data.csv', 'r') as f:
		for data_point in f:
			producer.send(topic, data_point.encode('utf-8'))
			sleep(data_rate)

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("-sid","--sensor_id", type=str, help="Sensor ID")
	parser.add_argument("-t","--topic", type=str, help="Topic on which the sensor will post, basically the node id of the worker", required=True)
	args = parser.parse_args()
	
	send_data(args.topic)