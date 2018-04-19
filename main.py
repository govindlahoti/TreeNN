import yaml
import sys
from node import Node
import time


def main():
	with open(sys.argv[1], 'r') as f:
		raw_data = yaml.load(f.read())

		default_push_interval = raw_data['default_push_interval']
		default_pull_interval = raw_data['default_pull_interval']
		default_delay = raw_data['default_delay']

		data = {}

		for x in raw_data['nodes']:
			data[x['id']] = x
			data[x['id']]['is_worker'] = True
			data[x['id']]['own_address'] = (x['ip'], x['port'])
			
			if 'pull_interval' not in x:
				x['pull_interval'] = default_pull_interval

			if 'push_interval' not in x:
				x['push_interval'] = default_push_interval


		for x in raw_data['nodes']:
			if 'parent_id' in x:
				data[x['id']]['parent_address'] = 'http://{}:{}'.format(data[x['parent_id']]['ip'], data[x['parent_id']]['port'])
				data[x['parent_id']]['is_worker'] = False
			else:
				data[x['id']]['parent_id'] = -1
				data[x['id']]['parent_address'] = None

		for x in data:
			data[x]['delays'] = {}
			for y in data:
				data[x]['delays'][y] = default_delay

		for x in raw_data['delays']:
			data[x['src_id']]['delays'][x['dest_id']] = x['delay']

		nodes = {}

		for x in data:
			node = Node(data[x])
		
		while True:
			time.sleep(9)



if __name__ == '__main__':
	main()