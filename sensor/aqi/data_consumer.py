import sys

from kafka import KafkaConsumer
consumer = KafkaConsumer('1')
# while True:
# 	data = consumer.poll(timeout_ms=100000, max_records=100)
# 	print(data)
# for msg in consumer:
# 	print(msg.value)
# 	if str(msg.value) == 'end':
# 		break
sensor_data, data_points = [], 0
for msg in consumer:
	if data_points==0: print(msg.value.decode('utf-8'))
	# sensor_data.append([msg.value.decode('utf-8').replace("\\", "")])
	data_points+=1
		# if data_points == self.window_size:
		# 	print(data_points,self.window_size)
		# 	break
	# return sensor_data