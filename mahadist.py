'''
Created on May 22, 2014
@author: Edward Sterkin
@email:  esterkin@umail.ucsb.edu
'''
import math
from sys import argv
import numpy as np
from numpy.linalg import inv

def get_array(filename):
	training_file = open(filename)
	M, N = map(int, training_file.readline().split())				#M = num rows, N = num dimensions 
	#print M, N
	training_data_array =  np.loadtxt(training_file) 				#load the data into numpy.ndarray			
	return training_data_array


def find_centroid(training_data_array):
	#get number of rows in array
	M = (training_data_array.shape)[0]	

	#sum up the columns 
	column_sums = training_data_array.sum(axis=0)	

	centroid = [x/M for x in column_sums]

	return centroid

def find_covariance(centroid, array):

	zero_mean_array = array - centroid #find zero mean matrix 
	k = (array.shape)[0]

	zero_mean_array_transpose = zero_mean_array.transpose()

	xxt = np.dot(zero_mean_array_transpose,zero_mean_array)
 
	covariance =  (1.0/k) * xxt 
	return covariance

def mahal(covariance, centroid, testdata):
	inv_covar = inv(covariance)
	i = 1
	for line in testdata:			#mahal = (xminusyT * E^T * xminusy)^0.5
		xminusy = line - centroid
		xminusyT = xminusy.transpose() #(X-Y)^T
		rhs = np.dot(xminusy, inv_covar)
		print "%d. %s -- %s" % (i,line, np.sqrt(np.dot(xminusyT, rhs)))  
		i+=1

if __name__ == '__main__':
    if len(argv) == 3:
	    traindata = get_array(argv[1])
	    testdata = get_array(argv[2])
	    centroid = find_centroid(traindata)
	    print 'Centroid:\n', centroid
	    covariance = find_covariance(centroid,traindata)
	    print 'Covariance matrix:\n', covariance
	    print 'Distances:'
	    mahal(covariance,centroid, testdata)
    else:
    	print 'Please input the testing file and the training file'
