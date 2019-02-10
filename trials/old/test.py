import csv
import network
import sys

EPOCHS = None
BATCH_SIZE = None
ETA = None
LAMBD = None
ALPHA = None
BETA = None

try:
    EPOCHS = int(sys.argv[1])
    BATCH_SIZE = int(sys.argv[2])
    ETA = float(sys.argv[3])
    LAMBD = float(sys.argv[4])
    ALPHA = float(sys.argv[5])
    BETA = float(sys.argv[6])
except:
    EPOCHS = 1
    BATCH_SIZE = 16
    ETA = 0.01
    LAMBD = 2
    ALPHA = 10
    BETA = 15

path = "../../data/"

train_data = []

for i in range(1,5):
    file = path+"Cluster" + str(i) + "_Data.csv"
    with open(file) as csv_file:
        csv_reader = csv.reader(csv_file)        
        train_data.extend(list(csv_reader))

input_length = 276
output_length = 48

net = network.Network([input_length, input_length, input_length, output_length])

args = {
    'epochs' : EPOCHS,
    'mini_batch_size' : BATCH_SIZE,
    'eta' : ETA,
    'lmbda' : LAMBD,
    'alpha' : ALPHA,
    'beta' : BETA
}

print(*sys.argv)

net.train(train_data,**args)

for i in range(1,10):
    file = path+"Cluster" + str(i) + "_Data.csv"
    with open(file) as csv_file:
        csv_reader = csv.reader(csv_file)        
        print(i,net.evaluate(list(csv_reader)))