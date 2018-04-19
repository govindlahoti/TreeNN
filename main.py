import yaml
import sys
from node import Node
import time


def main():
	with open(sys.argv[1], 'r') as f:
		raw_data = yaml.load(f.read())
		
		worker_push_interval = raw_data['worker_push_interval']
		worker_pull_interval = raw_data['worker_pull_interval']

		data = {}

		for x in raw_data['Nodes']:
			data[x['id']] = {}
			data[x['id']]['ip'] = x['ip']
			data[x['id']]['port'] = x['port']
			data[x['id']]['is_worker'] = True

		for x in raw_data['Nodes']:
			if 'parent_id' in x:
				data[x['id']]['parent_ip'] = data[x['parent_id']]['ip']
				data[x['id']]['parent_port'] = data[x['parent_id']]['port']
				data[x['parent_id']]['is_worker'] = False

		nodes = {}

		for x in data:
			y = data[x]
			if 'parent_ip' in y:
				parent_address = 'http://{}:{}'.format(y['parent_ip'], y['parent_port'])
			else:
				parent_address = None
			
			node = Node(x, (y['ip'], y['port']), parent_address, y['is_worker'], worker_push_interval, worker_pull_interval)
			nodes[x] = node

		time.sleep(9)



if __name__ == '__main__':
	main()