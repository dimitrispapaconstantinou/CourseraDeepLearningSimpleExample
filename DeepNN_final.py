#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 17:29:22 2017

@author: dimitris
"""

import numpy as np
import h5py
#import scipy.io # used for exporting arrays to matlab (amongst othersf)
 


np.random.seed(1)


def load_data():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes
####################################
    
def sigmoid(Z):
    A = 1/(1+np.exp(-Z))
    #cache=Z
    return A  #cache
    

def relu(Z):
    A = np.maximum(0,Z)        
#    cache = Z 
    return A #cache
##############################3





train_x_orig, train_y, test_x_orig, test_y, classes = load_data()
# Reshape the training and test examples 
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.


############# Global 
layers = [12288, 20, 7, 5, 1] #  4-layer model   (layer_dims)
#layers = [12288, 20, 20, 20, 1] #  4-layer model 
# The input layer is the 0 layer. So actually the above NN is a 4 layer network.
# 
#############






def init():
 
    np.random.seed(1)
    L=len(layers)
    parameters = {}
     
    for l in range(1, L):
        parameters['W'+str(l)]=np.random.randn(layers[l],layers[l-1])/ np.sqrt(layers[l-1]) # here was *0.001
#        parameters['W'+str(l)]=np.random.randn(layers[l],layers[l-1]) *np.sqrt(2/layers[l-1]) # He initialization
        parameters['b'+str(l)]=np.zeros((layers[l],1) )
#        print (l)
#        print("W"+str(l)+str( parameters['W'+str(l)].shape))
#        print("b"+str(l)+str( parameters['b'+str(l)].shape))
    return parameters
############################


def forwardPropagation(X, parameters ):
    # inputs
    # X the original matrix of the data. A big matrix with all the pictures.
    
    # returns a list (caches ), with length the number ot layers.
    # The layers are numbered from 0 to L. 
    # zero is the the first layer (the input layer)
    # the list contains a tuple (A, W b, Z) for each layer.
    caches=[]
    c=[]

    # parameters are in the form of W1, b1, W2, b2, W3, b3, W4, b4 ....
    # so the division by 2 shows indirectly the  number of layers.
    L = len(parameters)//2 #L=4 
    
    A=X # A0=X
    Atemp=[]

    # all the layers but the last use the RELU function 
    for l in range(1,L): # 1 2 3 , if number of layers is 4 ( starting from 0 = input layer)
        A_prev=A
        W=parameters['W'+str(l)]
        b=parameters['b'+str(l)]
        Z = np.dot(W,A_prev)+b
        A=relu(Z)
        c=(A,W,b,Z)         # create a tuple for the parametrs if the current layer
        caches.append(c)    # add the tuple to the caches list
        Atemp=A


    # calulate the last layer with the sigmoid function
    W=parameters['W'+str(L)]
    b=parameters['b'+str(L)]   
    Z = np.dot(W, Atemp)+b
    A_for_lastLayer=sigmoid(Z)
    c =(A_for_lastLayer,W, b, Z)
    caches.append(c)

#        caches=[(A,W,b,Z),     (A,W,b,Z),   (A,W,b,Z),  ...   (A,W,b,Z)]
#               first layer,   sec layer,   third layer, ...   L layer
#numbering        0              1             2                3           (indexing in the implemetation)
#               A1,w1,b1,z1    A2,W2b3,Z2      A3,W3,b3,Z3     A4,W4,b4,Z4      layers
    return caches
##############################################


def costFunction(forwardCaches, Y, lambd):
    # theloume to last element toy c pou periexei ta parameters apo to 
    # teleftaio layer   
    
    # input: a list c = forwardCaches
    
    
    m = Y.shape[1] # number of samples
	
    n= len(forwardCaches) # n=number of layers
    layerParams = forwardCaches[n-1]   # get the params of the last layer (the -1 is for the offset of the index that starts from 0.)
    AL = layerParams[0]
    
    # cost function
    logprobs = np.multiply(np.log(AL),Y) + np.multiply(np.log(1-AL),(1-Y))
    cost = - np.sum(logprobs) / m 
    
    
    temp=0.0
    
    #L2 regularization
    L=len(layers)
    for i in range(L-1):
#        print(i)
        current_tuple = forwardCaches[i]
        w=current_tuple[1]
        temp+= (np.sum(np.square(w)))   
    
    L2=temp* (lambd/(2*m) )
    
    costRegulized = cost+L2
    
    return cost, costRegulized
####################################################


def backwardPropagation(forwardCaches, X, Y): # c[A, W, b, Z]
    # inputs
    # X: The original matrix of the samples.
    # Y: The class of the samples. If the picture is cat or not.
    # c: The forwardCaches


    cacheBachwards=[]
    m = Y.shape[1] # number of samples
    
    L= len(forwardCaches) ## number of layers L=4
#    print(L)###########debug
    
    
     # the -1 is to corrent the offset from the
     # indexing of the array (Starts at 0)
	 # 3 is the last layer in nubmberint 0 1 2 3
	 # in case if 4 layers, the command below is (A4, W4,b4, Z4)
    layerParams = forwardCaches[L-1]  
         
    AL = layerParams[0] # get the 1 element of the tuple (A ,W ,b ,Z) 
     
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL)) # derivative of cost with respect to AL
     
    Z = layerParams[3]  # get the 4th element of the tuple (A ,W ,b ,Z)
    W = layerParams[1]
    s = 1/(1+np.exp(-Z))
    dZ = dAL *s * (1-s)

    # get the previous layer for the A3 value (if we have 4 layers 0 1 2 3 4)
	## in case of 4 layers, the command below is (A3, W3,b3, Z3) 
    layerParams= forwardCaches[L-2] 
    A_prev =layerParams[0]# get the 1 element of the tuple (A ,W ,b ,Z)
    
    dW = (np.dot(dZ,A_prev.T))/m
#    print("dW-sigm" + str(dW.shape)) ########debug
    db = (np.sum(dZ, axis=1, keepdims = True))/m
    currentTuple = (dAL,dW,db, Z ) #create the new tuple. currentTuple
    cacheBachwards.append(currentTuple)    # layer 4 . position [0] 
    
    dA_prev = np.dot(W.T, dZ) # dA3 = np.dot(W4.T, dZ4)
     

    for l in reversed(range(L-1)): #gia L= 4 kanei 2,1,0
        #print(l)
        layerParams=forwardCaches[l]  
        Z=layerParams[3] # 
        W=layerParams[1]
        dZ = np.array(dA_prev, copy=True)
        dZ[Z <= 0] = 0
        
        #find A prev for dw db 
        # L = len(forwardCaches)
        if ((l-1) >=0):
            layerParams=forwardCaches[l-1]
            A_prev = layerParams[0]
            dW = (np.dot(dZ,A_prev.T))/m
            db = (np.sum(dZ, axis=1, keepdims = True))/m
            currentTuple = (dA_prev,dW,db, Z )
            cacheBachwards.append(currentTuple)  
            dA_prev = np.dot(W.T, dZ)
        else: # in this case A_prev = A0 = X
            A_prev=X
            dW = (np.dot(dZ,A_prev.T))/m
            db = (np.sum(dZ, axis=1, keepdims = True))/m
            currentTuple = (dA_prev,dW,db, Z )
            cacheBachwards.append(currentTuple)
            
 
    cacheBachwards.reverse()    

    return cacheBachwards 
####################################################



def updateParameters(forwardCaches, backwardCaches, lr): 
    L = len(forwardCaches)
    params ={}

    for l in range(L): # 0 1 2 3 
        layerParamsF = forwardCaches[l]
        layerParamsB = backwardCaches[l]
        W = layerParamsF[1]
        dW =layerParamsB[1]
        b = layerParamsF[2]
        db =layerParamsB[2]      
      
        W = W - lr*dW
        b = b - lr*db
           
        params['W'+str(l+1)]=W
        params['b'+str(l+1)]=b
    
    return params 
####################################################
    








def L_model(X, Y, iterations, learning_rate, print_cost=True):
    
    # list of costs
    costs = []  
    
    # same seed for testing 111
    np.random.seed(1)    
    
    # initialiazation of weights and biases.
    params = init()
        
    # number of iterations for Gradient descent
    for i in range(iterations):

#         Step 1. Forward propagation. Returns a list of tuples containing
#         the parameters of the forwardPropagation returns a returns a list that 
#         contains a tuple (A, W b, Z) for each layer.
#         caches=[(A,W,b,Z),     (A,W,b,Z),   (A,W,b,Z),  ...   (A,W,b,Z)]
#                first layer,   sec layer,   third layer, ...   L layer
       forwardCaches = forwardPropagation(X, params) 
      
        #Step 2. Calculate cost. (scalar)
       cost, costReg =costFunction(forwardCaches, Y, lambd=0.1)
       
       #Step 3. Backward propagation 
       backwardCaches = backwardPropagation(forwardCaches, X, Y)
       
       #step 4. Update parametes
       params = updateParameters(forwardCaches, backwardCaches, learning_rate)
       

       if print_cost and i % 100 == 0:
           print ("Cost after iteration %i: cost: %f cosrReg:%f" %(i,  cost, costReg))
       if print_cost and i % 100 == 0:
           costs.append(cost)
        
    return 
####################################################



# triain the model.
L_model(train_x,train_y, iterations=2500, learning_rate=0.0075 )

