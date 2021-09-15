"""
Author: Viswambhar Yasa

Test cases for the optimizers.
Every optimizer is tested with 2 test cases. One with list of ints and other for weights of a NN
The ones where there is no use of epsilon to evade division by zer0 - the predicted values are directly asserted to be equal to hand-calculated values
The ones where there is a use of epsilon to evade division by zero - the absolute difference between hand calculated and predicted values is asserted to be very less,almost negligible 
In all the cases the gradients are takes as list of arrays of ones of same shape and learning rate is 0.1
"""



#Import required packages and functions from other files
import pytest 
from optimizers import *
import numpy as np
from NN_architecture import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
plt.style.use('dark_background')

def test_SGD_list_of_ints():
    """
    test stochastic gradient descent optimizer(SGD) with learning rate 0.1.
    tested for: one step of gradient descent for parameters 'x' and gradients 'dx' = [1,1,1]
    tested with : Hand calculated values parameters based on formula = parameters - learning rate * gradients

    """
    #Instantiating SGD object with lr= 0.1
    opt = SGD(lr=0.1)
    x = [1,1,1]
    dx = [1,1,1]
    #calling optimizer to take descent step
    nx = opt(x,dx)

    assert np.array_equal(nx,[0.9,0.9,0.9])


def test_SGD_NN():
    """
    test stochastic gradient descent optimizer(SGD) with learning rate 0.1
    tested for : randomly initialized weight matrices(x) of a Neural Network with number of neurons:5 , number of layers:1, input dimension: 5, output dimension:6 and gradients(dx) equal to ones of same shape

    tested with :  Hand calculated values parameters based on formula new parameters = parameters - learning rate * gradients
    """
    #Instantiating SGD object with lr= 0.1
    opt = SGD(lr=0.1)
    #Instantiating  the Neural Network
    model = NeuralNetLSTM(5,0,5,6)
    x = model.get_weights()
    dx=[]
    for i in range(len(x)):
        #gradients of shape equal to shape of parameters and value 1
        dx.append(np.ones_like(x[i]))
    nx=[]
    #Calculating Descent Step
    for i in range(len(x)):
        temp = x[i] - 0.1*dx[i]
        nx.append(temp)
    #Calling optimizer
    nx_opt = opt(x,dx)
    print(nx)
    print(nx_opt)

    for i in range(len(x)):
        assert np.array_equal(nx[i](),nx_opt[i]())

def test_Momentum_list_of_ints():
    """
    test Momentum optimizer with learning rate 0.1 and gamma = 0.8 
    tested for : one step of gradient descent for parameters 'x' and gradients 'dx' = [1,1,1] and accumulation vector which is unique prospect of Momentum.
    tested with :  Hand calculated values parameters based on formulae : new a = gamma*a + gradient and  new parameters = parameters - learning rate * new a


    """
    #Instantiating Optimizer 
    opt = Momentum(3,lr=0.1)
    x = [1,1,1]
    dx = [1,1,1]
    #Calling Optimizer
    nx = opt(x,dx)

    assert np.array_equal(nx,[0.9,0.9,0.9]) and np.array_equal(opt.a,dx)


def test_Momentum_NN():
    """
    test Momentum optimizer with learning rate 0.1 and gamma = 0.8 
    tested for : randomly initialized weight matrices(x) of a Neural Network with number of neurons:5 , number of layers:1, input dimension: 5, output dimension:6 and gradients(dx) equal to ones of same shape
                 and accumulation vector(new a) which is unique prospect of momentum optimizer.

    tested with :  Hand calculated values parameters based on formulae : new a = gamma*a + gradient and  new parameters = parameters - learning rate * new a


    """   
    #Instantiate NN
    
    model = NeuralNetLSTM(5,0,5,6)
    #Get Parameters
    x = model.get_weights()
    #Instantiate optimizer
    opt = Momentum(len(x),lr=0.1)
    dx=[]
    for i in range(len(x)):
        #gradients of shape equal to shape of parameters and value 1
        dx.append(np.ones_like(x[i]))
    nx=[]
    for i in range(len(x)):
        temp = x[i] - 0.1*dx[i]
        nx.append(temp)
    nx_opt = opt(x,dx)
    print(nx)
    print(nx_opt)

    for i in range(len(x)):
        assert np.array_equal(nx[i](),nx_opt[i]()) and np.array_equal(opt.a,dx)


def test_Adagrad_list_of_ints():
    """
    test Adagrad optimizer with learning rate 0.1 
    tested for : one step of gradient descent for parameters 'x' and gradients 'dx' = [1,1,1] and runner vector(new runner) which is unique prospect of Adagrad
    tested with :  Hand calculated values parameters based on formulae : new runner = runner + gradient^2 and  new parameters = parameters - learning rate/sqrt(runner) * gradients


    """
    #Instantiate optimizer
    opt = Adagrad(3,lr=0.1)
    x = [1,1,1]
    dx = [1,1,1]
    #Calling optimizer
    nx = opt(x,dx)
    print(nx)

    assert np.array_equal(nx,[0.900000001, 0.900000001, 0.900000001]) and np.array_equal(opt.runner,dx)

def test_Adagrad_NN():
    """
    test Adagrad optimizer with learning rate 0.1
    tested for : randomly initialized weight matrices(x) of a Neural Network with number of neurons:5 , number of layers:1, input dimension: 5, output dimension:6 and gradients(dx) equal to ones of same shape
                 and runner vector(new runner) which is unique prospect of Adagrad optimizer.
    tested with : Hand calculated values parameters based on formulae : new runner = runner + gradient^2 and  new parameters = parameters - learning rate/sqrt(runner) * gradients

    """
    #Instantiate NN
    model = NeuralNetLSTM(5,0,5,6)
    x = model.get_weights()
    #Instantiate Optimizer
    opt = Adagrad(len(x),lr=0.1)
    dx=[]
    for i in range(len(x)):
        dx.append(np.ones_like(x[i]))
    nx=[]
    for i in range(len(x)):
        temp = x[i] - (0.1/(1+1e-8))*dx[i]
        nx.append(temp)
    #calling optimizer
    nx_opt = opt(x,dx)
    print(nx)
    print(nx_opt)

    for i in range(len(x)):
        assert np.array_equal(nx[i](),nx_opt[i]()) and np.array_equal(opt.runner,dx)
    
def test_Adam_list_of_ints():
    """
    test Adam optimizer with learning rate 0.1 , beta1(The exponential decay rate for the first moment estimates) = 0.9, beta2(The exponential decay rate for the second-moment estimates ) =0.999
    tested for : one step of gradient descent for parameters 'x' and gradients 'dx' = [1,1,1] and momentum and velocity vectors which are unique prospects of Adam.
    tested with :  Hand calculated values based on formulae:
                   new momentum = beta1*momentum + (1-beta1)*gradients
                   new velocity = beta2*velocity + (1-beta2)*gradients*gradients
                   accumulation = learning rate * sqrt(1-beta2**iteration)/(1-beta1**iteration)
                   new parameters = parameters - accumulation*new momentum/sqrt(new velocity)

    """
    #Instantiating the optimizer
    opt = Adam(3,lr=0.1)
    x = [1,1,1]
    dx = [1,1,1]
    #Calling the optimizer
    nx = opt(x,dx)
    print(nx)
    print(opt.momentum)
    print(opt.velocity)

    assert np.array_equal(nx,[0.9000000416227625, 0.9000000416227625, 0.9000000416227625]) and np.array_equal(opt.momentum,[0.09999999999999998, 0.09999999999999998, 0.09999999999999998])\
        and np.array_equal(opt.velocity,[0.0010000000000000009, 0.0010000000000000009, 0.0010000000000000009])

def test_Adam_NN():
    """
    test Adam optimizer with learning rate 0.1 , beta1(The exponential decay rate for the first moment estimates) = 0.9, beta2(The exponential decay rate for the second-moment estimates ) =0.999
    tested for : randomly initialized weight matrices(x) of a Neural Network with number of neurons:5 , number of layers:1, input dimension: 5, output dimension:6 and gradients(dx) equal to ones of same shape
                 and momentum and velocity vectors which are unique prospects of Adam.
    tested with : Hand calculated values based on formulae:
                   new momentum = beta1*momentum + (1-beta1)*gradients
                   new velocity = beta2*velocity + (1-beta2)*gradients*gradients
                   accumulation = learning rate * sqrt(1-beta2**iteration)/(1-beta1**iteration)
                   new parameters = parameters - accumulation*new momentum/sqrt(new velocity)
    """
    #instantiating NN 
    model = NeuralNetLSTM(5,0,5,6)
    x = model.get_weights()
    #instantiating optimizer
    opt = Adam(len(x),lr=0.1)
    dx=[]
    for i in range(len(x)):
        dx.append(np.ones_like(x[i]))
    nx=[]
    for i in range(len(x)):
        #gradients of same shape and value 1
        temp = x[i] - 0.03162277343940645*(0.1/np.sqrt((0.001+1e-8)))*dx[i]
        nx.append(temp())
    #Calling optimizer
    nx_temp = opt(x,dx)
    nx_opt = [i() for i in nx_temp]
    print(nx[0])
    print(nx_opt[0])
    m = [0.1*i for i in dx]
    v = [0.001*i for i in dx]
    for i in range(len(x)):
        assert np.all(np.less(np.abs(m[i]-opt.momentum[i]),0.000000000001)) and np.all(np.less(np.abs(v[i]-opt.velocity[i]),0.000000000001)) \
            and np.all(np.less(np.abs(nx[i]-nx_opt[i]),1e-6))



def test_RMSProp_list_of_ints():
    """
    test RMSProp optimizer with learning rate 0.1 and decay rate 0.9 
    tested for : one step of gradient descent for parameters 'x' and gradients 'dx' = [1,1,1] and runner vector which is unique prospect of RMSProp
    tested with : Hand calculated values based on formula
                  new runner = runner * decay rate + (1-decay rate)*gradients**2
                  new parameters = parameters - learning rate/sqrt(runner)*gradients


    """
    #Instantiating optimizer
    opt = RMSProp(3,lr=0.1)
    x = [1,1,1]
    dx = [1,1,1]
    #Calling optimizer
    nx = opt(x,dx)
    print(opt.runner)


    assert np.array_equal(nx,[0.6837722439831617, 0.6837722439831617, 0.6837722439831617]) and np.array_equal(opt.runner,[0.09999999999999998, 0.09999999999999998, 0.09999999999999998])


def test_RMSProp_NN():
    """
    test RMSProp optimizer with learning rate 0.1 and decay rate 0.9 
    tested for : randomly initialized weight matrices(x) of a Neural Network with number of neurons:5 , number of layers:1, input dimension: 5, output dimension:6 and gradients(dx) equal to ones of same shape
                 and runner vector which is unique prospect of RMSprop
    tested with: tested with : Hand calculated values based on formula
                 new runner = runner * decay rate + (1-decay rate)*gradients**2
                 new parameters = parameters - learning rate/sqrt(runner)*gradients

    """
    #Instantiating NN
    
    model = NeuralNetLSTM(5,0,5,6)
    x = model.get_weights()
    #Instantiating optimizer
    opt = RMSProp(len(x),lr=0.1)
    dx=[]
    for i in range(len(x)):
        #gradients of same shape and value 1
        dx.append(np.ones_like(x[i]))
    nx=[]
    for i in range(len(x)):
        temp = x[i] - (0.1/np.sqrt((0.1)))*dx[i]
        nx.append(temp)
    nx_opt = opt(x,dx)
    print(nx[0]())
    print(nx_opt[0]())

    for i in range(len(x)):
        assert np.all(np.less(np.abs(nx[i]()-nx_opt[i]()),1e-8)) and 0.1-opt.runner[i]< 1e-15



def test_Adamax_list_ints():
    """
    test Adamax optimizer with learning rate 0.1 , beta1(The exponential decay rate for the first moment estimates) = 0.9, beta2(The exponential decay rate for the second-moment estimates ) =0.999
    tested for : one step of gradient descent for parameters 'x' and gradients 'dx' = [1,1,1] and momentum and velocity vectors which are unique prospects of Adamax
    tested with : Hand calculated values based on formulae:
                  new momentum = beta1*momentum + (1-beta1)*gradients
                  new velocity = max(velocity*beta2,normal value of gradients)
                  mhat =new momentum/(1-beta1**iteration)
                  new parameters= parameters - learning rate * mhat / new velocity
    """
    #instantiating optimizer
    opt = Adamax(3,lr=0.1)
    x = [1,1,1]
    dx = [1,1,1]
    #calling optimizer
    nx = opt(x,dx)
    print(nx)
    print(opt.momentum)
    print(opt.velocity)

    assert np.array_equal(nx,[0.900000001, 0.900000001, 0.900000001]) and np.array_equal(opt.momentum,[0.09999999999999998, 0.09999999999999998, 0.09999999999999998])\
        and np.array_equal(opt.velocity,[1,1,1])


def test_Adamax_NN():
    """
    test Adamax optimizer with learning rate 0.1 , beta1(The exponential decay rate for the first moment estimates) = 0.9, beta2(The exponential decay rate for the second-moment estimates ) =0.999
    tested for : randomly initialized weight matrices(x) of a Neural Network with number of neurons:5 , number of layers:1, input dimension: 5, output dimension:6 and gradients(dx) equal to ones of same shape
                 and momentum and velocity vectors which are unique prospects of Adamax.
    tested with : Hand calculated values based on formulae:
                  new momentum = beta1*momentum + (1-beta1)*gradients
                  new velocity = max(velocity*beta2,normal value of gradients)
                  mhat =new momentum/(1-beta1**iteration)
                  new parameters= parameters - learning rate * mhat / new velocity
    """
    #Instantiating NN
    model = NeuralNetLSTM(5,0,5,6)
    x = model.get_weights()
    #Instantiating Optimizer
    opt = Adamax(len(x),lr=0.1)
    dx=[]
    for i in range(len(x)):
        #Gradients of same shape and value 1
        dx.append(np.ones_like(x[i]))
    nx=[]
    for i in range(len(x)):
        temp = x[i] -(0.1/np.sqrt(1+1e-8))*dx[i]
        nx.append(temp())
    #Calling optimizer
    nx_temp = opt(x,dx)
    nx_opt = [i() for i in nx_temp]
    print(nx[0])
    print(nx_opt[0])
    m = [0.1*i for i in dx]
    v = dx
    for i in range(len(x)):
        assert np.all(np.less(np.abs(m[i]-opt.momentum[i]),0.0000000000000001)) and np.all(np.less(np.abs(v[i]-opt.velocity[i]),0.0000000000000001)) \
            and np.all(np.less(np.abs(nx[i]-nx_opt[i]),1e-8))



