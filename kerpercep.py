'''
Created on May 22, 2014
@author: Edward Sterkin
@email:  esterkin@umail.ucsb.edu
'''

import numpy as np
from sys import argv
from numpy import linalg

def get_array(filename):
	datafile = open(filename)
	datafile.readline()				#skip header 
	return np.loadtxt(datafile) 

#definition of the Gaussian kernel function 

def g_kernel(x, y, sigma):			
    return np.exp(-linalg.norm(x-y)**2 / (sigma ** 2) )


# generate the Gram matrix 
# @param X - training_x

def genGramMatrix(X,sigma):
	num_samples = (X.shape)[0]
	G = np.zeros((num_samples, num_samples))
	for i in range(num_samples):
		for j in range(num_samples):
			G[i,j] = g_kernel(X[i], X[j], sigma)
	return G

# generate the Gram matrix 
# @param G - gram matrix
# @param y - y output 

def genAlpha(G,y):
	num_samples = (G.shape)[0]
	alpha = np.zeros(num_samples) 
	converged = False
	while not converged:
		converged = True
		for i in range(num_samples):
			if np.sign(np.sum(G[:,i] * alpha * y)) != y[i]: 
				#for each sample 
				alpha[i] += 1
				converged = False
	return alpha


 # classify the test data

 # @param X - testing data
 # @param Alpha - alpha values matrix
 # @param sigma - sigma value 
 # @param fi_training_x - filtered input training data
 # @param fi_training_y - filtered output training data 

 
def classify(X,Alpha,y,sigma,fi_training_x,fi_training_y): 
	num_samples = (X.shape)[0] #rows
	#matrix to hold the predictions
	predictions = np.zeros(num_samples) 
	for i in range(num_samples):
		total = 0          
		for a, y, x in zip(Alpha, fi_training_y, fi_training_x):
			#iterate over the 3 lists in parallel 
			total += a * y * g_kernel(X[i],x,sigma)	
		predictions[i] = total
	return np.sign(predictions)

if __name__ == "__main__":

	if len(argv) == 6:
		#get data from stdin 

		sigma = float(argv[1])

		training_pos = get_array(argv[2])
		training_neg = get_array(argv[3])

		testing_pos = get_array(argv[4])
		testing_neg = get_array(argv[5])

		#create y-values for positive and negative class 
		training_y_pos = np.ones(len(training_pos)) * 1
		training_y_neg = np.ones(len(training_neg)) * -1

		testing_y_pos = np.ones(len(testing_pos)) * 1
		testing_y_neg = np.ones(len(testing_neg)) * -1

		#combine positive and negative training & testing data 
		#into one array
		#positives will be above the negatives 
		training_x = np.vstack((training_pos, training_neg)) 
		testing_x  = np.vstack((testing_pos, testing_neg))

		training_y = np.hstack((training_y_pos, training_y_neg))
		testing_y = np.hstack((testing_y_pos, testing_y_neg))

		#train the model

		gramMatrix = genGramMatrix(training_x,sigma)
		# print gramMatrix 

		alpha = genAlpha(gramMatrix,training_y)

		#filter data for non-zero values

		fi_alpha = alpha[alpha > 0]
		fi_training_x = training_x[alpha > 0]
		fi_training_y = training_y[alpha > 0]

		predictions = classify(testing_x, fi_alpha,training_y,sigma,fi_training_x,fi_training_y)

		# false positives && false negatives  
		num_fp = 0
		num_fn = 0
		for predicted, actual in zip(predictions, testing_y):
			if actual == -1 and predicted == 1:
				num_fp +=1
			if actual == 1 and predicted == -1:
				num_fn +=1
		error_rate =  (1.0 * (num_fp + num_fn)/len(testing_y)) * 100 

		print 'Alphas:', alpha
		print 'False positives:',num_fp
		print 'False negatives:',num_fn
		print 'Error rate', error_rate,'%'

	else: 
		print 'Please input the testing files and the training files'
