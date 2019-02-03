import csv
import network
import sys

EPOCHS = int(sys.argv[1])
BATCH_SIZE = int(sys.argv[2])
ETA = float(sys.argv[3])
LAMBD = float(sys.argv[4])
ALPHA = float(sys.argv[5])
BETA = float(sys.argv[6])

# code for storing grid neighbors in the dictionary
file_1  = "../../data/Cluster1_Data.csv"
file_2  = "../../data/Cluster2_Data.csv"
file_3  = "../../data/Cluster3_Data.csv"
file_4  = "../../data/Cluster4_Data.csv"
file_5  = "../../data/Cluster5_Data.csv"
file_6  = "../../data/Cluster6_Data.csv"
file_7  = "../../data/Cluster7_Data.csv"
file_8  = "../../data/Cluster8_Data.csv"
file_9  = "../../data/Cluster9_Data.csv"

train_data = []
test_data = []

for i in range(1,8):
    file = "../../data/Cluster" + str(i) + "_Data.csv"
    with open(file) as csv_file:
        csv_reader = csv.reader(csv_file)        
        train_data.extend(list(csv_reader))
	
for i in range(8,10):
    file = "../../data/Cluster" + str(i) + "_Data.csv"
    with open(file) as csv_file:
        csv_reader = csv.reader(csv_file)        
        test_data.extend(list(csv_reader))

input_length = 276
output_length = 48

net_new = network.Network([input_length, input_length, input_length, output_length])

args = {
    'epochs' : EPOCHS,
    'mini_batch_size' : BATCH_SIZE,
    'eta' : ETA,
    'lmbda' : LAMBD,
    'alpha' : ALPHA,
    'beta' : BETA
}

print(*sys.argv)
net_new.train(train_data,**args)
print(net_new.evaluate(train_data),net_new.evaluate(test_data))