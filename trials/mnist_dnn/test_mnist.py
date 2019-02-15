import csv
import mnist

def readMNIST(f):
	with open(f,'r') as p:
		return [list(map(int,rec)) for rec in csv.reader(p, delimiter=',')]

if __name__ == '__main__':

	for i in range(10):
		print(i)
		train_data = readMNIST('../mnist_data/train/biased-%d.csv'%i)
		
		kwargs = {
			'input_size'		: 784,
			'output_size'		: 10,
			'num_hidden_layers'	: 3,
			'hidden_layer_sizes': [20,10,10],
			'alpha'				: 0.4,
			'batch_size'		: 20,
			'epochs'			: 10
		}

		nn = mnist.MNISTModel(**kwargs)

		nn.train(train_data)

		for i in range(10):
			test_data = readMNIST('../mnist_data/test/%d.csv'%i)
			pred, acc = nn.evaluate(test_data)
			print('  ',i,acc)