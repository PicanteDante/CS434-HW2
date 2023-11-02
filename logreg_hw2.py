"""
Things I changed:
step_size = 0.002 | 0.0025
max_iters = 20000
dummy_aug = column of 1s
"""



import numpy as np
np.random.seed(42)
import matplotlib.pyplot as plt
import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

# GLOBAL PARAMETERS FOR STOCHASTIC GRADIENT DESCENT
step_size=0.0001
max_iters=1000

def main():

	# Load the training data
	logging.info("Loading data")
	X_train, y_train, X_test = loadData()

	logging.info("\n---------------------------------------------------------------------------\n")
	
	# Fit a logistic regression model on train and plot its losses
	logging.info("Training logistic regression model (No Bias Term)")
	w, losses = trainLogistic(X_train,y_train)
	y_pred_train = X_train@w >= 0

	logging.info("Learned weight vector: {}".format([np.round(a,4)[0] for a in w]))
	logging.info("Train accuracy: {:.4}%".format(np.mean(y_pred_train == y_train)*100))

	logging.info("\n---------------------------------------------------------------------------\n")
	
	X_train_bias = dummyAugment(X_train)
	X_test_bias = dummyAugment(X_test)
	sizes = []
	accuracies = []
	weights = []
	lossess = []
	
	# Fit a logistic regression model on train and plot its losses
	logging.info("Training logistic regression model (Added Bias Term)")
	for i in range(3001):
		ss = i * 0.00001
		
		#bias_w, bias_losses = trainLogistic(X_train_bias,y_train)
		#y_pred_train = X_train_bias@bias_w >= 0
		sizes.append(ss)
		
		bias_w, bias_losses = trainLogistic(X_train_bias,y_train, step_size=ss)
		y_pred_train = X_train_bias@bias_w >= 0
		
		weights.append(bias_w)
		lossess.append(bias_losses)
		accuracies.append(np.mean(y_pred_train == y_train)*100)
		
		print("StepSize: {:.4f}\tAccuracy: {:.5f}%\tWeights: {}".format(ss, accuracies[i], bias_w))
		
		#logging.info("Learned weight vector: {}".format([np.round(a,4)[0] for a in bias_w]))
		#logging.info("Train accuracy: {:.4}%".format(np.mean(y_pred_train == y_train)*100))
	
	'''
	# Fit a logistic regression model on train and plot its losses
	logging.info("Training logistic regression model (Added Two Bias Terms)")
	bias_w2, bias_losses2 = trainLogistic(X_train_bias2,y_train)
	y_pred_train = X_train_bias@bias_w >= 0

	logging.info("Learned weight vector: {}".format([np.round(a,4)[0] for a in bias_w2]))
	logging.info("Train accuracy: {:.4}%".format(np.mean(y_pred_train == y_train)*100))'''

	plt.figure(figsize=(16,7))
	plt.plot(sizes,accuracies)
	plt.xlabel("step_size")
	plt.ylabel("accuracy")
	plt.show()
	
	test_out = np.concatenate(np.array(sizes),
								np.array(accuracies),
								np.array(weights))
	header = np.array([["size", "accuracy", "weights"]])
	test_out = np.concatenate((header, test_out))
	np.savetxt('bigtests.csv', test_out, fmt='%s', delimiter=',')

	'''
	plt.figure(figsize=(16,7))
	plt.plot(range(len(losses)), losses, label="No Bias Term Added")
	plt.plot(range(len(bias_losses)), bias_losses, label="Bias Term Added")
	#plt.plot(range(len(bias_losses2)), bias_losses2, label="2 Bias Terms Added")
	plt.title("Logistic Regression Training Curve")
	plt.xlabel("Epoch")
	plt.ylabel("Negative Log Likelihood")
	plt.legend()
	plt.show()'''

	logging.info("\n---------------------------------------------------------------------------\n")
	
	logging.info("Running cross-fold validation for bias case:")
	
	# Perform k-fold cross
	'''
	for k in [2,3,4, 5, 10, 20, 50]:
		cv_acc, cv_std = kFoldCrossVal(X_train_bias, y_train, k)
		logging.info("{}-fold Cross Val Accuracy -- Mean (stdev): {:.4}% ({:.4}%)".format(k,cv_acc*100, cv_std*100))
	'''
	####################################################
	# Write the code to make your test submission here
	####################################################
	
	
	#	Actually make predictions
	pred_test_y = predict(X_test_bias, bias_w, 0.5)
	
	test_out = np.concatenate((np.expand_dims(np.array(range(pred_test_y.shape[0]),dtype=int), axis=1), pred_test_y), axis=1)
	header = np.array([["id", "type"]])
	test_out = np.concatenate((header, test_out))
	np.savetxt('test_predicted.csv', test_out, fmt='%s', delimiter=',')



######################################################################
# Q3.1 logistic 
######################################################################
# Given an input vector z, return a vector of the outputs of a logistic
# function applied to each input value
#
# Input: 
#   z --   a n-by-1 vector
#
# Output:
#   logit_z --  a n-by-1 vector where logit_z[i] is the result of 
#               applying the logistic function to z[i]
######################################################################
def logistic(z):
	logit_z = 1 / (1 + np.exp(-z))
	return logit_z


######################################################################
# Q3.2 calculateNegativeLogLikelihood 
######################################################################
# Given an input data matrix X, label vector y, and weight vector w
# compute the negative log likelihood of a logistic regression model
# using w on the data defined by X and y
#
# Input: 
#   X --   a n-by-d matrix of examples where each row
#                   corresponds to a single d-dimensional example
#
#   y --    a n-by-1 vector representing the labels of the examples in X
#
#   w --    a d-by-1 weight bector
#
# Output:
#   nll --  the value of the negative log-likelihood
######################################################################
def calculateNegativeLogLikelihood(X,y,w):
	z = X @ w
	logit_z = logistic(z)
	
	#sse = np.sum((y - logit_z) ** 2)
	#sae = np.sum(np.abs(y - logit_z))
	nll = -np.sum(y * np.log(logit_z + 0.0000001) + (1 - y) * np.log(1 - logit_z + 0.0000001))
	return nll




######################################################################
# Q4 trainLogistic
######################################################################
# Given an input data matrix X, label vector y, maximum number of 
# iterations max_iters, and step size step_size -- run max_iters of 
# gradient descent with a step size of step_size to optimize a weight
# vector that minimizies negative log-likelihood on the data defined
# by X and y
#
# Input: 
#   X --   a n-by-d matrix of examples where each row
#                   corresponds to a single d-dimensional example
#
#   y --    a n-by-1 vector representing the labels of the examples in X
#
#   max_iters --   the maximum number of gradient descent iterations
#
#   step_size -- the step size (or learning rate) for gradient descent
#
# Output:
#   w --  the d-by-1 weight vector at the end of training
#
#   losses -- a list of negative log-likelihood values for each iteration
######################################################################
def trainLogistic(X,y, max_iters=20000, step_size=0.002):
	# Initialize our weights with zeros
	w = np.zeros( (X.shape[1],1) )

	# Keep track of losses for plotting
	losses = [calculateNegativeLogLikelihood(X,y,w)]

	# Take up to max_iters steps of gradient descent
	for i in range(max_iters):

			   
		# Todo: Compute the gradient over the dataset and store in w_grad
		# .
		# . Implement equation 9.
		
		w_grad = np.dot(X.T, (logistic(X @ w) - y))
		
		# This is here to make sure your gradient is the right shape
		assert(w_grad.shape == (X.shape[1],1))
		
		# Take the update step in gradient descent
		w = w - step_size*w_grad

		# Calculate the negative log-likelihood with the
		# new weight vector and store it for plotting later
		losses.append(calculateNegativeLogLikelihood(X,y,w))

	return w, losses


######################################################################
# Q5 dummyAugment
######################################################################
# Given an input data matrix X, add a column of ones to the left-hand
# side
#
# Input: 
#   X --   a n-by-d matrix of examples where each row
#                   corresponds to a single d-dimensional example
#
# Output:
#   aug_X --  a n-by-(d+1) matrix of examples where each row
#                   corresponds to a single d-dimensional example
#                   where the the first column is all ones
#
######################################################################
def dummyAugment(X):
	aug_X = np.hstack((np.ones((X.shape[0], 1)), X))
	return aug_X


##################################################################
# Instructor Provided Code, Don't need to modify but should read
##################################################################

# Given a matrix X (n x d) and y (n x 1), perform k fold cross val.
def kFoldCrossVal(X, y, k):
  fold_size = int(np.ceil(len(X)/k))
  
  rand_inds = np.random.permutation(len(X))
  X = X[rand_inds]
  y = y[rand_inds]

  acc = []
  inds = np.arange(len(X))
  for j in range(k):
    
    start = min(len(X),fold_size*j)
    end = min(len(X),fold_size*(j+1))
    test_idx = np.arange(start, end)
    train_idx = np.concatenate( [np.arange(0,start), np.arange(end, len(X))] )
    if len(test_idx) < 2:
      break

    X_fold_test = X[test_idx]
    y_fold_test = y[test_idx]
    
    X_fold_train = X[train_idx]
    y_fold_train = y[train_idx]

    w, losses = trainLogistic(X_fold_train, y_fold_train)

    acc.append(np.mean((X_fold_test@w >= 0) == y_fold_test))

  return np.mean(acc), np.std(acc)


def predict(X_test, w, t):
    probabilities = logistic(X_test @ w)
    predictions = (probabilities >= t).astype(int)
    
    return predictions
    #return np.array(predicted_y,dtype=int)[:,np.newaxis]

# Loads the train and test splits, passes back x/y for train and just x for test
def loadData():
  train = np.loadtxt("train_cancer.csv", delimiter=",")
  test = np.loadtxt("test_cancer_pub.csv", delimiter=",")
  
  X_train = train[:, 0:-1]
  y_train = train[:, -1]
  X_test = test
  
  return X_train, y_train[:, np.newaxis], X_test   # The np.newaxis trick changes it from a (n,) matrix to a (n,1) matrix.


main()
