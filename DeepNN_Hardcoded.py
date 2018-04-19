#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 17:29:22 2017

@author: dimitris
"""

import numpy as np
import h5py
 


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


#######Global 
layers = [12288, 20, 7, 5, 1] #  4-layer model   (layer_dims)
#layers = [12288, 20, 20, 20, 1] #  4-layer model 
# The input layer is the 0 layer. So actually the above NN is a 4 layer network.
#############


def init():
    np.random.seed(1)
    L=len(layers)
    parameters = {}
     
    for l in range(1, L):
#        print(l)
        parameters['W'+str(l)]=np.random.randn(layers[l],layers[l-1])/ np.sqrt(layers[l-1])
        parameters['b'+str(l)]=np.zeros((layers[l],1) )
#        print (l)
#        print("W"+str(l)+str( parameters['W'+str(l)].shape))
#        print("b"+str(l)+str( parameters['b'+str(l)].shape))
    return parameters
############################

def fprop(X, parameters):
# X the original matrix of the data. A big matrix with all the pictures.

    L = len(parameters)//2

    
    A_List=[]
    W_List=[]
    b_List=[]
    Z_List=[]
    
    A0=X
    A_List.append(A0)
    
    W1 = parameters['W1']
    b1 = parameters['b1']
    Z1 = np.dot(W1,A0)+b1
    A1 = relu(Z1)
    A_List.append(A1)
    W_List.append(W1)
    b_List.append(b1)
    Z_List.append(Z1)

    
    W2 = parameters['W2']
    b2 = parameters['b2']
    Z2 = np.dot(W2,A1)+b2
    A2 = relu(Z2)
    A_List.append(A2)
    W_List.append(W2)
    b_List.append(b2)
    Z_List.append(Z2)

    
    W3 = parameters['W3']
    b3 = parameters['b3']
    Z3 = np.dot(W3,A2)+b3
    A3 = relu(Z3)
    A_List.append(A3)
    W_List.append(W3)
    b_List.append(b3)
    Z_List.append(Z3)

    
    W4 = parameters['W4']
    b4 = parameters['b4']
    Z4 = np.dot(W4,A3)+b4
    A4 = sigmoid(Z4)
    A_List.append(A4)
    W_List.append(W4)
    b_List.append(b4)
    Z_List.append(Z4)
    
    
#    print("fprop W1: "+str (W1.shape))######debug
#    print("fprop Z1: "+str (Z1.shape))######debug
#    print("fprop A1: "+str (A1.shape))######debug
#    
#    print("fprop W2: "+str (W2.shape))######debug
#    print("fprop Z2: "+str (Z2.shape))######debug
#    print("fprop A2: "+str (A2.shape))######debug
#    
#    print("fprop W3: "+str (W3.shape))######debug
#    print("fprop Z3: "+str (Z3.shape))######debug
#    print("fprop A3: "+str (A3.shape))######debug
#    
#    print("fprop W4: "+str (W4.shape))######debug 
#    print("fprop Z4: "+str (Z4.shape))######debug
#    print("fprop A4: "+str (A4.shape))######debug
#    print(Z4.shape)
#    print(W4)
#    
##    print(A_List.shape)
   
    cacheList=[]
    cacheList.append(A_List)
    cacheList.append(W_List)
    cacheList.append(b_List)
    cacheList.append(Z_List)

    return cacheList

###########################
    

def ccost(c,Y):
    A_List = c[0]
    
    L= len(A_List) # length = 5 positions
    #print(L)
    
    m=A_List[0].shape[1]
#    print(m)
    AL = A_List[L-1] # last position L-1 = 5-1=4
    
    logprobs = np.multiply(np.log(AL),Y) + np.multiply(np.log(1-AL),(1-Y))
    cost = - np.sum(logprobs) / m 
#    cost=1
    return cost

def bprop(c,Y):
    dA_List=[]
    dW_List=[]
    db_List=[]
    dZ_List=[]
    
    A_List = c[0]
    W_List= c[1]
    b_List = c[2]
    Z_List = c[3]
#    print(len(Z_List))  #####################DEBUG
   
    
    L= len(A_List)
    AL = A_List[L-1]
    m=A_List[0].shape[1]
    
    # Initializing the backpropagation
    ### START CODE HERE ### (1 line of code)
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL)) # derivative of cost with respect to AL
#    print(dAL.shape)
#    print(len(dA_List))
    ### END CODE HERE ###
    
    dA_List.append(dAL)
        
    s = sigmoid(dAL)
    dZ4 = dAL *s * (1-s)
    A3= A_List[3]
    W4 = W_List[len(W_List)-1]
    dW4 = (np.dot(dZ4,A3.T))/m
    
    db4 = (np.sum(dZ4, axis=1, keepdims = True))/m
    dA3 = np.dot(W4.T, dZ4)
    dA_List.append(dA3)
    dW_List.append(dW4)
    db_List.append(db4)
    dZ_List.append(dZ4)
    
    
    
    dZ3 = np.array(dA3, copy=True)
    Z3=Z_List[2]
    dZ3[Z3 <= 0] = 0
#    print(dZ3.shape)
#    print(Z3.shape)
    A2 = A_List[2]
    W3=  W_List[len(W_List)-2]
    dW3 = (np.dot(dZ3,A2.T))/m
    db3 = (np.sum(dZ3, axis=1, keepdims = True))/m
    dA2 = np.dot(W3.T, dZ3)
    
    dA_List.append(dA2)
    dW_List.append(dW3)
    db_List.append(db3)
    dZ_List.append(dZ3)
    
    
    
    
    dZ2 = np.array(dA2, copy=True)
    Z2=Z_List[1]
    dZ2[Z2 <= 0] = 0
    A1 = A_List[1]
    W2=  W_List[len(W_List)-3]
    dW2 = (np.dot(dZ2,A1.T))/m
    db2 = (np.sum(dZ2, axis=1, keepdims = True))/m
    dA1 = np.dot(W2.T, dZ2)
    
    dA_List.append(dA1)
    dW_List.append(dW2)
    db_List.append(db2)
    dZ_List.append(dZ2)
    
    
    dZ1 = np.array(dA1, copy=True)
    Z1=Z_List[0]
    dZ1[Z1 <= 0] = 0
    A0 = A_List[0]
    W1=  W_List[len(W_List)-4]
    dW1 = (np.dot(dZ1,A0.T))/m
    db1 = (np.sum(dZ1, axis=1, keepdims = True))/m
    dA0 = np.dot(W1.T, dZ1)
    
    dA_List.append(dA0)
    dW_List.append(dW1)
    db_List.append(db1)
    dZ_List.append(dZ1)
    
    
#    print("back prop W4"+ str(W4.shape))######debug
#    print("back prop dZ4"+ str(dZ4.shape))######debug
#    print("back prop dW4"+ str(dW4.shape))######debug
#    print("back prop dA3"+ str(dA3.shape))######debug
#    print("------------------------------")
#    
#    print("back prop W3"+ str(W3.shape))######debug
#    print("back prop dZ3"+ str(dZ3.shape))######debug
#    print("back prop dW3"+ str(dW3.shape))######debug
#    print("back prop dA3"+ str(dA2.shape))######debug
#    print("------------------------------")
#    
#    print("back prop W2"+ str(W2.shape))######debug
#    print("back prop dZ2"+ str(dZ2.shape))######debug
#    print("back prop dW2"+ str(dW2.shape))######debug
#    print("back prop dA2"+ str(dA1.shape))######debug
#    print("------------------------------")
#    
#    print("back prop W1"+ str(W1.shape))######debug
#    print("back prop dZ1"+ str(dZ1.shape))######debug
#    print("back prop dW1"+ str(dW1.shape))######debug
#    print("back prop dA0"+ str(dA0.shape))######debug
#    print("------------------------------")
#    print(len(dA_List)) ######debug
    
    
    cache2List=[]
    cache2List.append(dA_List)
    cache2List.append(dW_List)
    cache2List.append(db_List)
    cache2List.append(dZ_List)

    return cache2List

###################

def update(c, c2, lr):
    
    A_List = c[0]
    W_List = c[1]
    b_List = c[2]
    Z_List = c[3]


    dA_List = c2[0]
    dW_List = c2[1] 
    db_List = c2[2]
    dZ_List = c2[3]
    
    W_List2=[]
    b_List2=[]
    
#    print(len(W_List)) ##############DEBUG

    W1 = W_List[0]
    dW1 = dW_List[3]
    W1 = W1 - lr*dW1
    W_List2.append(W1)
    
    
    b1 = b_List[0]
    db1 = db_List[3]
    b1 = b1 - lr*db1
    b_List2.append(b1)

    
    
    W2 = W_List[1]
    dW2 = dW_List[2]
    W2 = W2 - lr*dW2
    W_List2.append(W2)
    
    b2 = b_List[1]
    db2 = db_List[2]
    b2 = b2 - lr*db2
    b_List2.append(b2)
    
    
    W3 = W_List[2]
    dW3 = dW_List[1]
    W3 = W3 - lr*dW3
    W_List2.append(W3)
    
    b3 = b_List[2]
    db3 = db_List[1]
    b3 = b3 - lr*db3
    b_List2.append(b3)
    
    W4 = W_List[3]
#    print("w4 before update "+str(W4))######debug
    dW4 = dW_List[0]
#    print("dw4              "+str(dW4))######debug
    W4 = W4 - lr*dW4
#    print("w4 after  update "+str(W4))######debug
    W_List2.append(W4)
    
    b4 = b_List[3]
    db4 = db_List[0]
    b4 = b4 - lr*db4
    b_List2.append(b4)
    
#    c[1]=W_List
#    c[2] = b_List
    
    cachList=[]
    cachList.append(A_List)
    cachList.append(W_List2)
    cachList.append(b_List2)
    cachList.append(Z_List)


    return cachList

############################
def L_model(X,Y, iterations, lr, print_cost=True):
    
    costs = []  
    
    np.random.seed(1)
    parameters = init()
    
    c=[]
    
    for i in range(iterations):
       c = fprop(X,parameters)
#       print("type c=" +str(type(c)))
       
       cost= ccost(c,Y)
#       print("type cost=" +str(type(cost)))
       
       c2=bprop(c,Y)       
#       print("type c2=" +str(type(c2)))
       
       
       c3 = update(c, c2, lr)
#       print("type c3=" +str(type(c3)))
       
       
       
       Wc3=c3[1]
       parameters['W1']=Wc3[0]
       parameters['W2']=Wc3[1]
       parameters['W3']=Wc3[2]
       parameters['W4']=Wc3[3]
       
       bc3 = c3[2]
       parameters['b1']=bc3[0]
       parameters['b2']=bc3[1]
       parameters['b3']=bc3[2]
       parameters['b4']=bc3[3]
       

       
       
       
       
#       print ("Cost after iteration %i: %f" %(i, cost))
#       print("---")
       if print_cost and i % 100 == 0:
           print ("Cost after iteration %i: %f" %(i, cost))
       if print_cost and i % 100 == 0:
           costs.append(cost)
        
#    return final w,b,A2
    
#    a = c[0]
#    w= c[1]
#    b=c[2]
#    z=c[3]
#    print(str(len(a)))
#    print(str(len(w)))
#    print(str(len(b)))
#    print(str(len(z)))
    return 





L_model(train_x,train_y, iterations=2500, lr=0.009 )

