import sys
import csv 
from scipy import stats
import pickle

def mapToFloat(x):
	x = x[1:-1]
	x = x.replace("'", "")
	x = tuple(x.split(','))
	z = []
	for w in x:
		temp = ()
		if w:
		   if(w != ' '):
			   z.append(w)
	z = list(z)       
	r = map(float,z)
	return list(r)

def mapToFloaty(y):
	y = y[1:-2]
	y = y.replace("'", "")
	y = tuple(y.split(','))
	z = []
	for x in y:
		temp = ()
		if x:
		   if(x != '  '):
			   z.append(x) 
	z = list(z)
	r = map(float,z)
	return list(r)

if __name__ == "__main__":

	filename = sys.argv[1]

	with open(filename) as csv_file:
		csv_reader = csv.reader(csv_file)
		train_data = list(csv_reader)

	data = []

	for x,xs1,xs2,xt1,xt2,y,flag in train_data:
		x = mapToFloat(x)
		xs1 = mapToFloat(xs1)
		xs2 = mapToFloat(xs2)
		xt1 = mapToFloat(xt1)
		xt2 = mapToFloat(xt2)
		y = mapToFloaty(y)

		data.append(x+xs1+xs2+xt1+xt2+y)
	kernel = stats.gaussian_kde(data)
	print(kernel)

	out = open(filename+'.pickle','wb')
	pickle.dump(kernel,out)
	out.close()

	pick_kernel = pickle.load(open(filename+'.pickle',"rb"))
	print(pick_kernel.resample(len(data[0])))
	

# # The logic for generating data
# np.random.seed(42)
# data_trend = np.random.normal(0, 1, (48, 729))

# def get_data(filename,random=False):
# 	"""Returns the randomly generated data"""
# 	def get_y(x):
# 		return sigmoid(np.dot(data_trend, x))
	
# 	if not random:
# 		np.random.seed(42)
	
# 	x_vals = [np.random.normal(0, 1, (729, 1)) for _ in range(20)]
# 	y_vals = map(get_y, x_vals)

# 	return zip(x_vals, y_vals)