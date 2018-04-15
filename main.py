import yaml
import sys
from node import Node

def main():
	with open(sys.argv[1], 'r') as f:
		raw_data = yaml.load(f.read())
		
		data = {}

		for x in raw_data:
			data[raw_data['id']] = {}
			data[raw_data['id']]['ip'] = x['ip']
			data[raw_data['id']]['port'] = x['port']

		for x in raw_data:
			if 'parent_id' in x:
				data[raw_data['id']]['parent_ip'] = data[x['parent_id']]['ip']
				data[raw_data['id']]['parent_port'] = data[x['parent_id']]['port']

		nodes = {}

		for x in data:
			node = Node(x['id'], (x['ip'], x['port']), 'http://{}:{}'.format(x['parent_ip'], x['parent_port']))
			nodes[x['id']] = node


		print nodes['2'].a
		print nodes['1'].a
		print nodes['1'].push_to_parent(2,3)
		print nodes['1'].pull_from_parent()
		print nodes['2'].a
		print nodes['2'].a



if __name__ == '__main__':
	main()