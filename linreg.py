'''
Created on May 22, 2014
@author: Edward Sterkin
@email:  esterkin@umail.ucsb.edu
'''
import math
from sys import argv
import numpy as np
np.set_printoptions(threshold=np.nan)

def get_array(filename):
	datafile = open(filename)
	datafile.readline()				#skip header 
	return np.loadtxt(datafile) 

if __name__ == "__main__":

	if len(argv) == 3:
		#get data from stdin 

		training_data = get_array(argv[1])
		testing_data_orig = get_array(argv[2])	
		testing_data = testing_data_orig

		last_column = (training_data.shape)[1]-1 #index of last column 
		
		training_y = training_data[:,last_column]
		training_x = training_data[:,:last_column]
		
		#train the equation 

		training_x = training_x.transpose() 
		#get num of samples
		n = np.max(training_x.shape)
		training_x = np.vstack([np.ones(n), training_x]).T


		training_y =  training_y.reshape((n,1))
		xtx = np.dot(training_x.T, training_x)
		xtx_inverse = np.linalg.inv(xtx)

		#get the weights
		w = np.dot(np.dot(xtx_inverse, training_x.T), training_y) 	

		n = np.max(testing_data.shape)

		#if single var linear regression
		if testing_data.shape == (n,):
			testing_data = np.vstack([np.ones(n), testing_data]).T
				
		diff =  (w.shape[0] - testing_data.shape[1])
		ones = np.ones([testing_data.shape[0], diff])
		testing_data = np.hstack([ ones, testing_data])	
		
		predict_y = np.dot(testing_data, w)
		
		print 'w,t = ', w
		count = 1
		for x, y in zip(testing_data_orig, predict_y):
			print "%d. %s -- %s" % (count,x,y)  
			count+=1

	else: 
		print 'Please input the testing file and the training file'
