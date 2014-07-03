'''
Created on May 25, 2014
@author: Edward Sterkin
@email:  esterkin@umail.ucsb.edu
'''
import math
from sys import argv
from collections import OrderedDict #available in Python 2.7+
import numpy as np
np.set_printoptions(threshold=np.nan)

def get_array(filename):
	datafile = open(filename)
	datafile.readline()				#skip header 
	return np.loadtxt(datafile) 

if __name__ == "__main__":

	if len(argv) == 4:
		#get data from stdin 

		k = int(argv[1])
		training_data = get_array(argv[2])	
		testing_data = get_array(argv[3])	

		last_column = (training_data.shape)[1]-1 #index of last column

		num_test_samples = (testing_data.shape)[0]
		num_train_samples = (training_data.shape)[0]
		num_train_features = (training_data.shape)[1]-1 #don't include classification
		# print training_data.shape

		#for each test sample
		for i in range(num_test_samples):
			Z = np.zeros([num_train_samples,num_train_features+2])
			for j in range(num_train_samples):
				training_row = training_data[j,:last_column]
				testing_row = testing_data[i]
				distance = np.linalg.norm(training_row-testing_row)
				Z[j] = np.append(training_data[j], distance)

			#sort Z by distance
			Z = Z[Z[:,num_train_features+1].argsort()]
			Z = Z[0:k,:] #get first k rows of Z
			classes = (Z[:,num_train_features]).tolist() #extract the classes column (and convert to list)
			
			freq = OrderedDict() #preserve insertion order

			for c in classes:
				if c in freq:
	   				freq[c] += 1
				else:
					freq[c] = 1
			most = 0

			for key, val in freq.items():
				if val > most:
					most = val
					most_frequent_class = key

			print i+1, testing_data[i], '--', int(most_frequent_class)

	else: 
		print 'Please input the testfile and the training file'
