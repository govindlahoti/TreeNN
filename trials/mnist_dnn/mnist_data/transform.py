import csv
import numpy as np
import matplotlib.pyplot as plt

from mlxtend.data import loadlocal_mnist

## load mnist
def save_mnist():
	X, y = loadlocal_mnist(
	        images_path='train-images.idx3-ubyte', 
	        labels_path='train-labels.idx1-ubyte')
	y = y.reshape(len(y),1)

	X = np.where(X>0, 1, 0)

	data = np.concatenate((X,y), axis=1)
	np.savetxt(fname='data.csv', X=data, delimiter=',', fmt='%d')

def save_test_mnist():
	X, y = loadlocal_mnist(
	        images_path='t10k-images.idx3-ubyte', 
	        labels_path='t10k-labels.idx1-ubyte')
	y = y.reshape(len(y),1)

	X = np.where(X>0, 1, 0)
	
	data = np.concatenate((X,y), axis=1)
	np.savetxt(fname='test_data.csv', X=data, delimiter=',', fmt='%d')

	save_digits('test_data.csv','test/')

## Visualize data
def visualize(fname):
	with open(fname, 'r') as csv_file:
	    for data in csv.reader(csv_file):
	        label = data[-1]
	        pixels = data[:-1]

	        pixels = np.array(pixels, dtype='uint8')
	        pixels = pixels.reshape((28, 28))

	        plt.title('Label is {label}'.format(label=label))
	        plt.imshow(pixels, cmap='gray')
	        plt.show()

	        break 

## Save digitwise data
def save_digits(f,path='digits/'):
	digits = { i:[] for i in range(10)}
	with open(f, 'r') as csv_file:
		for data in csv.reader(csv_file):
			label = int(data[-1])
			x = [int(i) for i in data]
			digits[label].append(x)

	for k in digits:
		np.savetxt(fname='%s%d.csv'%(path,k), X=np.array(digits[k]), delimiter=',', fmt='%d')

def generate_biased_dataset(f,datapoints):
	digits = { i:[] for i in range(10)}
	with open(f, 'r') as csv_file:
		for data in csv.reader(csv_file):
			label = int(data[-1])
			x = [int(i) for i in data]
			digits[label].append(x)

	for k in digits:
		prob = np.array([64 if i==k else 4 for i in range(10)]) ## 64% images are digit k
		indices = np.random.choice(range(10),(datapoints,),p=prob/100)
		d = [ digits[i][np.random.choice(len(digits[i]),)] for i in indices]
		# print(np.bincount(indices))

		np.savetxt(fname='train/biased-%d.csv'%k, X=d, delimiter=',', fmt='%d')

if __name__ == '__main__':
	np.random.seed(42)
	save_mnist()
	save_test_mnist()
	visualize('data.csv')
	save_digits('data.csv')
	generate_biased_dataset('data.csv',6000)
