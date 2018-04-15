import yaml
import sys
from node import Node

def main():
	with open(sys.argv[1], 'r') as f:
		raw_data = yaml.load(f.read())['Nodes']
		
		data = {}

		for x in raw_data:
			data[x['id']] = {}
			data[x['id']]['ip'] = x['ip']
			data[x['id']]['port'] = x['port']

		for x in raw_data:
			if 'parent_id' in x:
				data[x['id']]['parent_ip'] = data[x['parent_id']]['ip']
				data[x['id']]['parent_port'] = data[x['parent_id']]['port']

		nodes = {}

		for x in data:
			y = data[x]
			if 'parent_ip' in y:
				parent_address = 'http://{}:{}'.format(y['parent_ip'], y['parent_port'])
			else:
				parent_address = None
			
			node = Node(x, (y['ip'], y['port']), parent_address)
			nodes[x] = node


		print nodes[2].a
		print nodes[1].a
		print nodes[1].push_to_parent(2,3)
		print nodes[1].pull_from_parent()
		print nodes[2].a
		print nodes[2].a



if __name__ == '__main__':
	main()