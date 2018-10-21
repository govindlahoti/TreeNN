'''
https://stackoverflow.com/questions/22130548/extracting-a-random-line-in-a-file-without-loading-the-file-into-ram-in-python
'''

import os
import heapq
import psutil
import random

py = psutil.Process(os.getpid())
SIZE = 10
with open('../data/Cluster8_Data.csv') as fin:
	sample = heapq.nlargest(SIZE, fin, key=lambda L: random.random())
print(py.memory_percent())