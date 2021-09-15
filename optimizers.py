"""
Author: Viswambhar Yasa

Optimizers for training the Neural Networks are defined here.
Inheritance is used to define all the optimizers as children of parent class "optimizer"
"""




#Import required packages and classes from other files.

import numpy as np
import autodiff as ad
from NN_architecture_2 import NeuralNetLSTM
import matplotlib.pyplot as plt 

class optimizer():
    """
    Base class /Parent class for all optimizers
    """
    def _forward_pass(self):
        """
        common polymorphic method for all optimizers.
        "_" denotes that it is private (shouldn't be accessed directly)
        and defining it here with raising error enables it to be overriden in child classes.  
        """
        raise NotImplementedError
    def __call__(self,params,grad_params):
        """
        This call invokes the private forward class method.
        Input arguments:
        params: The parameters which are being passed to the optimizers as list
        grad_params: The gradients of parameters(in the same order) which are being passed to the optimizer as a list
        returns: new parameters after forward pass based on optimization algorithm.  
        """
        new_params = self._forward_pass(params,grad_params)
        return new_params




class SGD(optimizer):
    def __init__(self,lr=0.00146):
        """
        Initializer for Stochastic Gradient descent optimizer
        Arguments: 
        lr : The learning rate with which the gradient step should be taken(integer/float), default is 0.00146  

        """
        self.lr = lr
    def _forward_pass(self,params,grad_params):
        """
        Takes a descent step by calculating the new parameters according to SGD update rule :
        new parameters = parameters - learning rate * gradients
        Inputs:
        params: The parameters which are being passed to the optimizers as list
        grad_params: The gradients of parameters(in the same order) which are being passed to the optimizer as a list
        returns: new parameters after descent step  
        
        """
        if isinstance(params,int):
            n=1
        else:
            n = len(params)
        new_params= []
        for i in range(n):
            new_params.append(params[i] - self.lr*grad_params[i])
        return new_params


class Momentum(optimizer):
    def __init__(self,num_params,lr=0.00146,gamma=0.8):
        """
        Initializer for momentum optimizer
        inputs:
        num_params: number of parameters which are ought to be passed
        lr: The learning rate with which the gradient step should be taken(integer/float), default is 0.00146  
        gamma: The momentum hyperparameter that accelerates the convergence towards the relevant direction and reduces the fluctuation to the irrelevant direction
                default is 0.9 

        """
        self.lr = lr
        self.num_params = num_params
        self.gamma = gamma
        self.num_params = num_params
        self.a = [0 for _ in range(num_params)]

    def _forward_pass(self,params,grad_params):
        """
        Takes a descent step by calculating the new parameters according to Momentum update rule :
        new a = gamma*a + gradient 
        new parameters = parameters - learning rate * new a
        Inputs:
        params: The parameters which are being passed to the optimizers as list
        grad_params: The gradients of parameters(in the same order) which are being passed to the optimizer as a list
        returns: new parameters after descent step  

        """
        new_params=[]
        for i in range(self.num_params):
            self.a[i] = self.gamma * self.a[i] + grad_params[i]
            new_params.append(params[i]- self.lr * self.a[i])
        return new_params
    

class Adagrad(optimizer):
    def __init__(self,num_params,lr=0.00146):
        """
        Initializer for Adagrad optimizer
        inputs:
        num params: number of parameters which are ought to be passed
        lr: The learning rate with which the gradient step should be taken(integer/float), default is 0.00146  
        """
        self.lr = lr
        self.num_params = num_params
        self.runner = [0 for _ in range(num_params)]
    
    def _forward_pass(self,params,grad_params):
        """
        Takes a descent step by calculating the new parameters according to Adagrad update rule :
        new runner = runner + gradient^2 
        new parameters = parameters - learning rate/sqrt(runner) * gradients
        Inputs:
        params: The parameters which are being passed to the optimizers as list
        grad_params: The gradients of parameters(in the same order) which are being passed to the optimizer as a list
        returns: new parameters after descent step  

        """
        new_params = []
        for i in range(self.num_params):
            self.runner[i] =  self.runner[i]  + grad_params[i]**2
            new_params.append(params[i] - self.lr/(np.sqrt(self.runner[i])+ 1e-8) * grad_params[i])


        return new_params


class Adam(optimizer):
    def __init__(self,num_params,lr=0.00146,b1=0.9,b2=0.999):
        """
        Initializer for Adam optimizer
        inputs:
        num params: number of parameters which are ought to be passed
        lr: The learning rate with which the gradient step should be taken(integer/float), default is 0.00146  
        b1 : The exponential decay rate for the first moment(integer/float), default is 0.9
        b2 : The exponential decay rate for the second moment(integer/float), default is 0.999
        """
        self.counter =0
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.num_params = num_params
        self.momentum = [0 for _ in range(num_params)]
        self.velocity = [0 for _ in range(num_params)]
    def _forward_pass(self,params,grad_params):
        """
        Takes a descent step by calculating the new parameters according to Adam update rule :
        new momentum = beta1*momentum + (1-beta1)*gradients
        new velocity = beta2*velocity + (1-beta2)*gradients*gradients
        accumulation = learning rate * sqrt(1-beta2**iteration)/(1-beta1**iteration)
        new parameters = parameters - accumulation*new momentum/sqrt(new velocity)

        Inputs:
        params: The parameters which are being passed to the optimizers as list
        grad_params: The gradients of parameters(in the same order) which are being passed to the optimizer as a list
        returns: new parameters after descent step 
        """
        new_params = []
        self.counter += 1 
        for i in range(self.num_params):
            self.momentum[i] = self.b1 * self.momentum[i]  + (1-self.b1)*grad_params[i]
            self.velocity[i] = self.b2 * self.velocity[i]  + (1-self.b2)*grad_params[i]**2
            accumulation = self.lr * np.sqrt(1 - self.b2 ** self.counter) / (1 - self.b1 ** self.counter + 1e-8)
            new_params.append(params[i]-accumulation * self.momentum[i] / (np.sqrt(self.velocity[i]) + 1e-8))
        
        
        return new_params



class RMSProp(optimizer):
    def __init__(self,num_params,lr=0.00146,decay_rate=0.9):
        """
        Initializer for RMSProp optimizer
        inputs:
        num params: number of parameters which are ought to be passed
        lr: The learning rate with which the gradient step should be taken(integer/float), default is 0.00146  
        decay_rate : decay rate/moving average hyper parameter (int/float), default is 0.9

        """
        self.num_params = num_params
        self.lr = lr
        self.decay_rate = decay_rate
        self.runner = [0 for _ in range(num_params)]
    def _forward_pass(self,params,grad_params):
        """
        Takes a descent step by calculating the new parameters according to RMSProp update rule :
        new runner = runner * decay rate + (1-decay rate)*gradients**2
        new parameters = parameters - learning rate/sqrt(runner)*gradients
        Inputs:
        params: The parameters which are being passed to the optimizers as list
        grad_params: The gradients of parameters(in the same order) which are being passed to the optimizer as a list
        returns: new parameters after descent step 

        """
        new_params = []
        for i in range(self.num_params):
            self.runner[i] =  self.decay_rate*self.runner[i]  + (1-self.decay_rate)*grad_params[i]**2
            new_params.append(params[i] - self.lr/(np.sqrt(self.runner[i])+ 1e-8) * grad_params[i])
        return new_params


class Adamax(optimizer):
    def __init__(self,num_params,lr=0.00146,b1=0.9,b2=0.99):
        """
        Initializer for Adamax optimizer
        Inputs:
        num params: number of parameters which are ought to be passed
        lr: The learning rate with which the gradient step should be taken(integer/float), default is 0.00146  
        b1 : The exponential decay rate for the first moment(integer/float), default is 0.9
        b2 : The exponential decay rate for the second moment(integer/float), default is 0.999
        """
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.num_params = num_params
        self.counter =0
        self.momentum = [0 for _ in range(num_params)]
        self.velocity = [0 for _ in range(num_params)]
    def _forward_pass(self,params,grad_params):
        """
        Takes a descent step by calculating the new parameters according to Adamax update rule :
        new momentum = beta1*momentum + (1-beta1)*gradients
        new velocity = max(velocity*beta2,normal value of gradients)
        mhat =new momentum/(1-beta1**iteration)
        new parameters= parameters - learning rate * mhat / new velocity
        Inputs:
        params: The parameters which are being passed to the optimizers as list
        grad_params: The gradients of parameters(in the same order) which are being passed to the optimizer as a list
        returns: new parameters after descent step 


        """
        self.counter += 1
        epsilon = 1e-8

        new_params=[]
        for i in range(self.num_params):
            self.momentum[i] = self.b1 * self.momentum[i]  + (1-self.b1)*grad_params[i]
            #print(self.momentum[i])
            self.velocity[i] = max(self.b2*self.velocity[i], abs(np.linalg.norm(grad_params[i])))

            
            #print(self.velocity[i])
            mhat = self.momentum[i]/(1-self.b1**self.counter)
            new_params.append(params[i] - (self.lr*mhat)/(self.velocity[i]+epsilon))
        return new_params
        


