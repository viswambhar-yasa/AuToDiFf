"""
Author: Viswambhar Yasa

This is the definition of base class for all the operations in this autodiff package
Every variable and operation is an extension of "Node" which is a point in a computational graph.

This node and first extension of node "variable" as starters of a compuational graph are defined here.

CITATION: These two classes are directly taken from "https://github.com/bgavran/autodiff"
"""




#Import required packages
import time
import numbers
import numpy as np
from contextlib import contextmanager


class Node:
    """
    Node is like  the blue print for every operation and variable to be defined in the autodiff package
    Every Node does two things:
    1.Calculates the value 
    2.Passes the information about gradient 
    Therefore it is like a wrapper around Numpy which uses it as a Numerical Kernel. 
    """
    #Some attributes common to all the NOdes
    epsilon = 1e-12
    id = 0
    context_list = []

    def __init__(self, children, name="Node"):
        """
        params:
        children : attributes which enter a node to be operated (eg: two numbers which need to be added)
        Node Id : to keep track of number of nodes instantiated to sort while collecting the derivatives
        cached: To check whether overriding has taken place
        shape: shape of the value of node if dealing with numpy arrays 
        """
        # wraps normal numbers into Variables
        self.children = [child if isinstance(child, Node) else Variable(child) for child in children]
        self.name = name
        self.cached = None
        self.shape = None

        self.context_list = Node.context_list.copy()
        self.id = Node.id
        Node.id += 1

    def _eval(self):
        """
        This is left unimplemented and _ is used in the function name(Virtual method) such that:
        1.It is a private method (Not accessed by the object directly)
        2.It is overriden by all the child classes according to their operation (Add class adds two numbers and so on..)

        :return: returns the value of the evaluated Node
        """
        raise NotImplementedError()

    def _partial_derivative(self, wrt, previous_grad):
        """
        
        Method which calculates the partial derivative of self with respect to the wrt Node.
        By defining this method without evaluation of any nodes, higher-order gradients
        are available for free.

        This is left unimplemented and _ is used in the function name(Virtual method) such that:
        1.It is a private method (Not accessed by the object directly)
        2.It is overriden by all the child classes according to their operation (derivative of x*y wrt x is y and so on..)

        :param wrt: instance of Node, partial derivativative with respect to it
        :param previous_grad: gradient with respect to self
        :return: an instance of Node whose evaluation yields the partial derivative
        """
        raise NotImplementedError()

    def eval(self):
        """
        The method to access the private _eval method. 
        Only accessed when the private method is appropriately overriden. 
        """
        #Sanity check
        if self.cached is None:
            self.cached = self._eval()

        return self.cached

    def partial_derivative(self, wrt, previous_grad):
        """
        The method to access the private _partial derivative method. 
        Only accessed when the private method is appropriately overriden. 
        """
        with add_context(self.name + " PD" + " wrt " + str(wrt)):
            return self._partial_derivative(wrt, previous_grad)

#Some magic methods defined to give a normalised look to the code(e.g: x+y looks better than Add(x,y))

    def __call__(self, *args, **kwargs):
        return self.eval()

    def __str__(self):
        return self.name  # + " " + str(self.id)

    def __add__(self, other):
        from .ops import Add
        return Add(self, other)

    def __neg__(self):
        from .ops import Negate
        return Negate(self)

    def __sub__(self, other):
        return self.__add__(other.__neg__())

    def __rsub__(self, other):
        return self.__neg__().__add__(other)

    def __mul__(self, other):
        from .ops import Mul
        return Mul(self, other)

    def __matmul__(self, other):
        from .high_level_ops import MatMul
        return MatMul(self, other)

    def __rmatmul__(self, other):
        from .high_level_ops import MatMul
        return MatMul(other, self)

    def __imatmul__(self, other):
        return self.__matmul__(other)

    def __truediv__(self, other):
        from .ops import Recipr
        return self.__mul__(Recipr(other))

    def __rtruediv__(self, other):
        from .ops import Recipr
        return Recipr(self).__mul__(other)

    def __pow__(self, power, modulo=None):
        from .ops import Pow
        return Pow(self, power)

    __rmul__ = __mul__
    __radd__ = __add__

    def __getitem__(self, item):
        from .reshape import Slice
        return Slice(self, item)


class Variable(Node):
    """
    Starter of all Computational Graphs because every operation starts with a set of variables
    Derivatives end here and the last calculated gradient is the total gradient of the compuational graph

    """

    def __init__(self, value, name=None):
        """
        If we forgot to name the variable, it is better to name it as the string of value itself.
        Rest initialized same as Super class-Node 
        params:
        value : The value to be stored in the variable
        stored as private attribute and can be accessed by only equivalent public method to keep sanctity.
        """

        if name is None:
            name = str(value)  # this op is really slow for np.arrays?!
        super().__init__([], name)


        if isinstance(value, numbers.Number):
            self._value = np.array(value, dtype=np.float64)
        else:
            self._value = value
        self.shape = self._value.shape
    
    #Decorators to calculate the value by accessing the private variable
    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, val):
        self.cached = self._value = val

    def _eval(self):
        """
        The overriden implementation
        """
        return self._value

    def _partial_derivative(self, wrt, previous_grad):
        """
        The overriden implementation
        """
        if self == wrt:
            return previous_grad
        return 0


@contextmanager
def add_context(ctx):
    Node.context_list.append(ctx + "_" + str(time.time()))
    try:
        yield
    finally:
        del Node.context_list[-1]
