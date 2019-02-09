import numpy as np
import csv
from datetime import datetime
from distlib.util import CSVReader
#import pandas as pd
import networkSeq
import new_network
import govind_network
import timeit

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
#f = open("test.txt",'w')

for i in range(8,9):
    file = "../../data/Cluster" + str(i) + "_Data.csv"
    #train_data = []
    with open(file) as csv_file:
        csv_reader = csv.reader(csv_file)
        
        train_data.extend(list(csv_reader))
        # print "file" , i, len(train_data[0]), len(train_data)
	

with open(file_8) as csv_file:
        csv_reader = csv.reader(csv_file)
        test_data = list(csv_reader)
        # print "file" , i, len(test_data[0]), len(test_data)

input_length = 276
output_length = 48

start = timeit.default_timer()
net = networkSeq.Network([input_length, input_length, input_length, output_length])  
net_new = new_network.Network([input_length, input_length, input_length, output_length])
net_govind = govind_network.Network([input_length, input_length, input_length, output_length])  
stop1 = timeit.default_timer()

EPOCHS = 10
BATCH_SIZE = 16
ETA = 0.01
LAMBD = 2
ALPHA = 10
BETA = 15

args = {
    'epochs' : EPOCHS,
    'mini_batch_size' : BATCH_SIZE,
    'eta' : ETA,
    'lmbda' : LAMBD,
    'alpha' : ALPHA,
    'beta' : BETA
}
# print("Govind")
# net_govind.SGD(train_data, EPOCHS, BATCH_SIZE, ETA)
# print("Alka")
# net.SGD(train_data, EPOCHS, BATCH_SIZE, ETA, LAMBD, ALPHA, BETA, test_data)
print("New")
net_new.train(train_data,**args)

# print(net.evaluate(test_data))
# print(net_new.evaluate_new(test_data))
# print(net_govind.evaluate(test_data))

stop2 = timeit.default_timer()
# print "initialization", (stop1-start), "SGD time", (stop2-stop1)
