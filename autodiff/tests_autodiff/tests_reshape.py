"""
Author: Viswambhar Yasa

The following test cases test the functinality of all the operations employed in the reshape section of the module i.efunctionalities which deal with the reshaping of arrays

Aim: All the test cases more or less aim for checking whether the reshaping function is working as expected (asserted aginst known values) and derivatives are flowing adequately
and all derivatives are hand calculated just like in Einsum.
No Anomalies have been reported 
WARNING:

Running these tests , all tests will pass but the following warning might be shown.

This update in syntax of python unfortunately came after implementation of the functionalities, and this won't actually come into force until the future versions of python (current- 3.7.5).  


FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.


"""


#Import required packages
import pytest
import numpy as np 
from autodiff.core.reshape import *
from autodiff.core.node import *
from autodiff.core.grad import grad



def test_concat_two_scalars():
    x = Variable(np.array([1]))
    y = Variable(np.array([1]))
    z = Concat(x,y)

    #z1 = Concat(x,y,1)
    
    assert isinstance(z,Concat) and np.array_equal(z(),np.array([1,1])) 
def test_concat_two_scalars1():
    x = Variable(np.array([[1]]))
    y = Variable(np.array([[1]]))
    z = Concat(x,y,0)
    #print(z())
    
   
    #z1 = Concat(x,y,1)
    assert isinstance(z,Concat) and np.array_equal(z(),np.array([[1],[1]])) 
def test_concat_two_scalars2():
    x = Variable(np.array([[1]]))
    y = Variable(np.array([[1]]))
    z = Concat(x,y,1)
    
    #z1 = Concat(x,y,1)
    assert isinstance(z,Concat) and np.array_equal(z(),np.array([[1,1]])) 

def test_concat_two_1DArray():
    x = Variable(np.array([0,1,2,3,4]))
    y = Variable(np.array([5,6,7,8]))
    z = Concat(x,y)
    

    #z1 = Concat(x,y,1)
    assert isinstance(z,Concat) and np.array_equal(z(),np.linspace(0,8,9)) 

def test_concat_two_1DArray1():
    x = Variable(np.array([[0,1,2,3,4]]))
    y = Variable(np.array([[5,6,7,8,9]]))
    z = Concat(x,y)
    

    #z1 = Concat(x,y,1)
    assert isinstance(z,Concat) and np.array_equal(z(),np.array([[0,1,2,3,4],[5,6,7,8,9]])) 

def test_concat_highD_arrays():
    x = Variable(np.random.rand(3,3,3))
    y = Variable(np.random.rand(3,3,3))
    z = Concat(x,y,0)
    z1= Concat(x,y,1)
    z2 = Concat(x,y,2)
    


    assert isinstance(z,Concat) and isinstance(z1,Concat) and isinstance(z2,Concat) \
        and np.array_equal(z(),np.concatenate((x(),y()),0)) and np.array_equal(z1(),np.concatenate((x(),y()),1)) and np.array_equal(z2(),np.concatenate((x(),y()),2))  \
            

def test_reshape_array():
    x = Variable(np.array([1,2,3,4,5,6]))
    y = Reshape(x,(2,3))
    dy = grad(y,[x])[0]
    assert isinstance(y,Reshape) and np.array_equal(y(),np.array([[1,2,3],[4,5,6]])) and np.array_equal(dy(),np.ones_like(x()))


def test_reshape_array_1():
    x = Variable(np.random.randn(3,3,4))
    y = Reshape(x,(4,3,3))
    y1 = Reshape(x,(36,))
    y2 = Reshape(x,(1,36))
    y3 = Reshape(x,(1,1,36))
    dydx = grad(y,[x])[0]
    dy1dx = grad(y1,[x])[0]
    dy2dx = grad(y2,[x])[0]
    print(dydx().shape)
    assert isinstance(y,Reshape) and np.array_equal(y(),np.reshape(x(),(4,3,3))) and np.array_equal(y1(),np.reshape(x(),(36,))) and np.array_equal(y2(),np.reshape(x(),(1,36))) \
        and np.array_equal(y3(),np.reshape(x(),(1,1,36))) and np.array_equal(dydx(),np.ones((3,3,4))) and np.array_equal(dy1dx(),np.ones((3,3,4))) and  np.array_equal(dy2dx(),np.ones((3,3,4))) 


def test_slice_array():
    x = np.linspace(0,10,11)
    X = Variable(x,"X")
    y = X[0]
    y1 = X[0:3]
    print(y1())
    dydx = grad(y,[X])[0]
    dy1dx = grad(y1,[X])[0]
    print(dydx())

    assert isinstance(y,Slice) and y()==0 and np.array_equal(y1(),np.array([0,1,2])) and np.array_equal(dydx(),[1,0,0,0,0,0,0,0,0,0,0]) and np.array_equal(dy1dx(),[1,1,1,0,0,0,0,0,0,0,0])

def test_slice_array_highD():
    x = np.random.rand(3,3,3)
    X = Variable(x,"X")
    y = X[0:2,1:2]
    y1 = X[:-2,1:2,0:1]
    #print(y1())
    dydx = grad(y,[X])[0]
    dy1dx = grad(y1,[X])[0]
    
    print(dydx())
    print(dy1dx())
    temp = np.zeros((3,3,3))
    temp[0:2,1:2] = np.ones_like(y())
    temp1 = np.zeros((3,3,3))
    temp1[:-2,1:2,0:1] = np.ones_like(y1())

    assert isinstance(y,Slice) and np.array_equal(y(),x[0:2,1:2]) and np.array_equal(y1(),x[:-2,1:2,0:1]) and np.array_equal(dydx(),temp) and np.array_equal(dy1dx(),temp1)

def test_pad_array():
    x = np.array([1,1,1,1])
    X = Variable(x,"X")
    y = Pad(X,5,4)
    
    
    assert isinstance(y,Pad) and np.array_equal(y(),np.array([4,4,4,4,4,1,1,1,1,4,4,4,4,4]))

def test_pad_array_high():
    x = np.random.rand(2,3)
    X = Variable(x,"X")
    y = Pad(X,[[1,0],[0,2]],[0,0])
    print(y())
    temp = np.zeros((3,5))
    temp[1:,0:3] = x


    assert isinstance(y,Pad) and np.array_equal(y(),temp)


