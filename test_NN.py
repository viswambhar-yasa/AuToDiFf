"""
Author: Viswambhar Yasa

Basically all the test cases below have same structure. 
Aim : To test the forward propagation of the Neural Network 
Expected : The calculations are done for a Neural Network with one hidden layer whose weights are set to 1.  
Obtained : The output of the Neural Network class 
Remarks: Each test case employs different set of inputs.
"""




import autodiff as ad 
import numpy as np
from NN_architecture import *

import pytest 

def Sigmoid(x):
        return 1/(1+np.exp(-x))

def test_sigmoid_is_true():
    x = 0
    y=0.5
    yf = Sigmoid(x)
    flag = np.array_equal(y,yf)
    assert flag == True

"""
def test_NN1_forward_prop_ones_init_is_true1():
    model = NeuralNetLSTM(2,0,2,1)
    wts = model.get_weights()
    model.set_weights([np.ones_like(i()) for i in  wts])
    X=np.array([[0.5,0.5]])
    W = np.ones((2,2))
    B = np.ones((1,2))
    S = Sigmoid(np.dot(X,W)  +  B )
    Z= Sigmoid(np.dot(X,W)+np.dot(S,W) + B)
    G = np.copy(Z)
    R = np.copy(G)
    H = Sigmoid(np.dot(X,W)+np.dot(S*R,W)+B)
    Snew = ((np.ones_like(G)-G)*H) + (Z*S)
    Wf = np.ones((2,1))
    Bf = np.ones(1)
    output = np.dot(Snew,Wf) + Bf
    print(output)
    print(model.output(X)())
    flag = np.array_equal(output,model.output(X)() )
    assert flag==True
"""
def test_NN1_forward_prop_ones_init_is_true2():
    model = NeuralNetLSTM(2,0,2,1)
    wts = model.get_weights()
    model.set_weights([np.ones_like(i()) for i in  wts])
    X=np.array([[0,0]])
    W = np.ones((2,2))
    B = np.ones((1,2))
    S = Sigmoid(np.dot(X,W)  +  B )
    Z= Sigmoid(np.dot(X,W)+np.dot(S,W) + B)
    G = np.copy(Z)
    R = np.copy(G)
    H = Sigmoid(np.dot(X,W)+np.dot(S*R,W)+B)
    Snew = ((np.ones_like(G)-G)*H) + (Z*S)
    Wf = np.ones((2,1))
    Bf = np.ones(1)
    output = np.dot(Snew,Wf) + Bf
    flag = np.array_equal(output,model.output(X)() )
    assert flag==True    


   
def test_NN1_forward_prop_ones_init_is_true4():
    model = NeuralNetLSTM(2,0,2,1)
    wts = model.get_weights()
    model.set_weights([np.ones_like(i()) for i in  wts])
    X=np.array([[-100.1894,-100.54]])
    W = np.ones((2,2))
    B = np.ones((1,2))
    S = Sigmoid(np.dot(X,W)  +  B )
    Z= Sigmoid(np.dot(X,W)+np.dot(S,W) + B)
    G = np.copy(Z)
    R = np.copy(G)
    H = Sigmoid(np.dot(X,W)+np.dot(S*R,W)+B)
    Snew = ((np.ones_like(G)-G)*H) + (Z*S)
    Wf = np.ones((2,1))
    Bf = np.ones(1)
    output = np.dot(Snew,Wf) + Bf
    flag = np.array_equal(output,model.output(X)() )
    assert flag==True 

def test_layer_NN1_ones_init_is_true1():
    X=np.array([[100,100]])
    W = np.ones((2,2))
    B = np.ones((1,2))
    S = Sigmoid(np.dot(X,W)  +  B )
    Z= Sigmoid(np.dot(X,W)+np.dot(S,W) + B)
    G = np.copy(Z)
    R = np.copy(G)
    H = Sigmoid(np.dot(X,W)+np.dot(S*R,W)+B)
    Snew = ((np.ones_like(G)-G)*H) + (Z*S)
    print(Snew)

    layer = lstm_layer(2,2)
    wts = layer.get_weights_layer()
    layer.set_params_layer([np.ones_like(i()) for i in  wts])
    S = ad.Variable(S,"S")
    X=ad.Variable(X,"X")
    output = layer.output_layer(S,X)
    print(output())
    flag = np.array_equal(output(),Snew)
    assert flag == True 


def test_layer_NN1_ones_init_is_true2():
    X=np.array([[-0.1548484,0.494]])
    W = np.ones((2,2))
    B = np.ones((1,2))
    S = Sigmoid(np.dot(X,W)  +  B )
    Z= Sigmoid(np.dot(X,W)+np.dot(S,W) + B)
    G = np.copy(Z)
    R = np.copy(G)
    H = Sigmoid(np.dot(X,W)+np.dot(S*R,W)+B)
    Snew = ((np.ones_like(G)-G)*H) + (Z*S)
    print(Snew)

    layer = lstm_layer(2,2)
    wts = layer.get_weights_layer()
    layer.set_params_layer([np.ones_like(i()) for i in  wts])
    S = ad.Variable(S,"S")
    X=ad.Variable(X,"X")
    output = layer.output_layer(S,X)
    print(output())
    flag = np.array_equal(output(),Snew)
    assert flag == True 


def test_layer_NN1_ones_init_is_true3():
    X=np.array([[10000,1000000]])
    W = np.ones((2,2))
    B = np.ones((1,2))
    S = Sigmoid(np.dot(X,W)  +  B )
    Z= Sigmoid(np.dot(X,W)+np.dot(S,W) + B)
    G = np.copy(Z)
    R = np.copy(G)
    H = Sigmoid(np.dot(X,W)+np.dot(S*R,W)+B)
    Snew = ((np.ones_like(G)-G)*H) + (Z*S)
    print(Snew)

    layer = lstm_layer(2,2)
    wts = layer.get_weights_layer()
    layer.set_params_layer([np.ones_like(i()) for i in  wts])
    S = ad.Variable(S,"S")
    X=ad.Variable(X,"X")
    output = layer.output_layer(S,X)
    print(output())
    flag = np.array_equal(output(),Snew)
    assert flag == True 


def test_layer_NN1_ones_init_is_true4():
    X=np.array([[616464516,45461616]])
    W = np.ones((2,2))
    B = np.ones((1,2))
    S = Sigmoid(np.dot(X,W)  +  B )
    Z= Sigmoid(np.dot(X,W)+np.dot(S,W) + B)
    G = np.copy(Z)
    R = np.copy(G)
    H = Sigmoid(np.dot(X,W)+np.dot(S*R,W)+B)
    Snew = ((np.ones_like(G)-G)*H) + (Z*S)
    print(Snew)

    layer = lstm_layer(2,2)
    wts = layer.get_weights_layer()
    layer.set_params_layer([np.ones_like(i()) for i in  wts])
    S = ad.Variable(S,"S")
    X=ad.Variable(X,"X")
    output = layer.output_layer(S,X)
    print(output())
    flag = np.array_equal(output(),Snew)
    assert flag == True 
