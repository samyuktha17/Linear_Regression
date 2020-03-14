"""
Do not change the input and output format.
If our script cannot run your code or the format is improper, your code will not be graded.

The only functions you need to implement in this template is linear_regression_noreg, linear_regression_invertible，regularized_linear_regression,
tune_lambda, test_error and mapping_data.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures 
###### Q1.1 ######
def mean_absolute_error(w, X, y):
    """
    Compute the mean absolute error on test set given X, y, and model parameter w.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing test feature.
    - y: A numpy array of shape (num_samples, ) containing test label
    - w: a numpy array of shape (D, )
    Returns:
    - err: the mean absolute error
    """
    #####################################################
    # TODO 1: Fill in your code here #
    #####################################################
  
    err=(1/len(X))*(np.sum(abs(np.dot(X,w)-y)))
    return err

###### Q1.2 ######
def linear_regression_noreg(X, y):
  """
  Compute the weight parameter given X and y.
  Inputs:
  - X: A numpy array of shape (num_samples, D) containing feature.
  - y: A numpy array of shape (num_samples, ) containing label
  Returns:
  - w: a numpy array of shape (D, )
  """
  #####################################################
  #	TODO 2: Fill in your code here #
  #####################################################		
  w = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(X),X)),np.transpose(X)),y)
  #print(np.shape(w))
  return w

###### Q1.3 ######
def linear_regression_invertible(X, y):
    """
    Compute the weight parameter given X and y.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing feature.
    - y: A numpy array of shape (num_samples, ) containing label
    Returns:
    - w: a numpy array of shape (D, )
    """
    #####################################################
    # TODO 3: Fill in your code here #
    #####################################################
    A = np.dot(np.transpose(X),X)
    non_invertible=False
    evalues, evectors = np.linalg.eig(A)
    for i in evalues:
        if abs(i)<pow(10,-5):
            non_invertible=True
            break
    l=0
    while non_invertible==True:
        l+=0.1
        flag=0
        inv=np.linalg.inv(A+(l*np.identity(len(A))))
        evalues, evectors = np.linalg.eig(inv)
        for i in evalues:
            if abs(i)<pow(10,-5):
                flag=1
                continue
        if flag==1:
            non_invertible=False
            break
    
    w = np.dot(inv,(np.dot(np.transpose(X),y)))
    return w


###### Q1.4 ######
def regularized_linear_regression(X, y, lambd):
    """
    Compute the weight parameter given X, y and lambda.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing feature.
    - y: A numpy array of shape (num_samples, ) containing label
    - lambd: a float number containing regularization strength
    Returns:
    - w: a numpy array of shape (D, )
    """
  #####################################################
  # TODO 4: Fill in your code here #
  #####################################################		
      
    A = np.dot(np.transpose(X),X)
    
    inv= np.linalg.inv(A+(lambd*np.identity(len(A))))
  
    w = np.dot(inv,(np.dot(np.transpose(X),y)))
    return w

###### Q1.5 ######
def tune_lambda(Xtrain, ytrain, Xval, yval):
    """
    Find the best lambda value.
    Inputs:
    - Xtrain: A numpy array of shape (num_training_samples, D) containing training feature.
    - ytrain: A numpy array of shape (num_training_samples, ) containing training label
    - Xval: A numpy array of shape (num_val_samples, D) containing validation feature.
    - yval: A numpy array of shape (num_val_samples, ) containing validation label
    Returns:
    - bestlambda: the best lambda you find in lambds
    """
    #####################################################
    # TODO 5: Fill in your code here #
    #####################################################		
    lambd=1e-19
    min_mae=None
    count=-19
    while lambd<=1e+19:
        w= regularized_linear_regression(Xtrain,ytrain,lambd)
        if min_mae==None or mean_absolute_error(w, Xval, yval)<min_mae:
            min_mae=mean_absolute_error(w,Xval,yval)
            bestlambda=lambd
        count=count+1
        lambd=round(lambd*10,abs(count))
        
    
    #bestlambda = None
    return bestlambda
    

###### Q1.6 ######
def mapping_data(X, power):
    """
    Mapping the data.
    Inputs:
    - X: A numpy array of shape (num_training_samples, D) containing training feature.
    - power: A integer that indicate the power in polynomial regression
    Returns:
    - X: mapped_X, You can manually calculate the size of X based on the power and original size of X
    """
    #####################################################
    # TODO 6: Fill in your code here #
    #####################################################		
    X_new=list()
    for x in X:
        x_new=list()
        for i in range(1,power+1):
            x_new.append(np.power(x,i))
        x_new=np.array(x_new)
        X_new.append(x_new.flatten())
    
    X=np.array(X_new)
    
    return X

