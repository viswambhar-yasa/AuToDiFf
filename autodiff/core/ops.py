"""
Author: Yasa Viswambhar

This file contains the operations which are going to be used in this Automatic Differentiation framework.
Every operation is a child of Node and overrides the virtual eval and partial derivative methods according to their implementation   
All the classes have the same outline .
->The __init__ method initializes the nodes and calls the parent __init__ method. 
->The partial derivative and eval method overrides the corresponding virtual methods according to their implementations 
CITATION: The classes Add,Mul,Einsum,function ReduceSumToShape are directly taken from "https://github.com/bgavran/autodiff"
"""


#Import required packages
import re
import numpy as np
import numbers
from .node import Node, Variable, add_context
from .reshape import ReduceSumKeepDims

from functools import reduce
from string import ascii_lowercase

#
def module_wrapper(fn):
    """
    defining a module wrapper to wrap in context
    helps to keep track of new nodes formed and partial derivatives calculation
    parameters:
    fn: function 
    returns a method which returns a functional value wrapped in context

    """
    def wrap_in_context(*args, **kwargs):
        with add_context(fn.__name__):
            return fn(*args, **kwargs)

    return wrap_in_context


def letters_from_tuple(tpl):
    """
    a small function to get the lower case letters for einsum
    params:
    tuple of letters
    returns lower case of letters 
    eg: length is 1 , we get 'a' ,2 'b' and so on


    """
    return ascii_lowercase[:len(tpl)]


def shape_from_elems(*elems):
    """
    function which broadcasts a list of elements into a single shape and gives that shape as output  
    params:
    elems: list of elements 
    returns broadcasted shape
    """
    if len(elems) == 0:
        return 1,
    return np.broadcast(*[np.ones(elem.shape) for elem in elems]).shape


@module_wrapper
def ReduceSumToShape(tensor, to_shape):
    """
    function which uses Reduce Sum keep Dims class from Reshape. 
    params:
    tensor: basically the array whose dimensions need  to be reduced 
    to_shape: required output shape 
    returns array with required shape 
    """
    if tensor.shape == to_shape:
        return tensor
    previous_grad_letters = letters_from_tuple(tensor.shape)
    if len(to_shape) == 0:
        wrt_letters = ""
    else:
        wrt_letters = previous_grad_letters[-len(to_shape):]  # take last letters of previous_grad_letters

    new_curr_grad = Einsum(str(previous_grad_letters) + "->" + str(wrt_letters), tensor)
    reduced_sum_grad = ReduceSumKeepDims(new_curr_grad, axes=[i for i, val in enumerate(to_shape) if val == 1])
    return reduced_sum_grad


class Add(Node):
    """
    Operation Add which adds two or more Nodes. 

    """
    def __init__(self, *elems, name="Add"):
        if not elems:
            name = "0-" + name
        super().__init__(list(elems), name)
        self.shape = shape_from_elems(*self.children)

    def _eval(self):
        # Using python sum instead of np.sum because python converts types correctly
        return np.array(sum([elem() for elem in self.children]))

    def _partial_derivative(self, wrt, previous_grad):
        # previous_grad will always be of shape of the shape of the "largest" variable
        # we need to sum across those other axes

        wrt_count = self.children.count(wrt)
        grad = previous_grad * Variable(wrt_count)
        return ReduceSumToShape(grad, wrt.shape)


class Mul(Node):
    """
    Operation which multiplies two nodes (a simple '*' equivalent)
    """
    fn = lambda x, y: x * y

    def __init__(self, *elems, name="Mul"):
        if not elems:
            name = "1-" + name
        super().__init__(list(elems), name)
        self.shape = shape_from_elems(*self.children)

    def _eval(self):
        # Mul broadcasts
        return reduce(Mul.fn, [child() for child in self.children], 1)

    def _partial_derivative(self, wrt, previous_grad):
        # previous_grad will always be of shape of the shape of the "largest" variable ?
        # we need to sum across those other axes ?
        add_list = []
        for loc, child in enumerate(self.children):
            if child == wrt:
                add_list.append(Mul(*[ch for i, ch in enumerate(self.children) if loc != i]))

        grad = previous_grad * Add(*add_list)
        return ReduceSumToShape(grad, wrt.shape)


class Negate(Node):
    """
    Operation which does Negation on a Node
    """
    def __init__(self, node, name="Negate"):
        super().__init__([node], name)
        self.node = self.children[0]
        self.shape = self.node.shape

    def _eval(self):
        return -self.node()

    def _partial_derivative(self, wrt, previous_grad):
        if self.node == wrt:
            return -previous_grad
        else:
            return 0


class Recipr(Node):
    """
    Elementwise reciprocal operation
    """

    def __init__(self, node, name="Reciprocal"):
        """
        Elementwise reciprocal

        """
        super().__init__([node], name)
        self.node = self.children[0]
        self.shape = self.node.shape

    def _eval(self):
        return 1 / (self.node() + Node.epsilon)

    def _partial_derivative(self, wrt, previous_grad):
        if self.node == wrt:
            return - previous_grad * self * self
        return 0


class Einsum(Node):
    """
    Einstein Summation operation: Instead of individually defining all the matrix,tensor and vector operations , it is elegant to define one EinSum of
    and add wrappers to all the known operations because Einsum encompasses all possible matrix, tensor , vector operations
    """
    def __init__(self, op_str, *operands, name="EinSum"):
        super().__init__(list(operands), name + " " + op_str)
        # TODO ellipsis currently can't be in the middle of op_letters!
        self.op_str = op_str
        self.operands = self.children

        self.opnames = re.split(",|->", self.op_str)
        self.all_letters = "".join(set("".join(self.opnames[:-1])))
        # can also be "..." to an arbitrary shape tuple
        self.letter_to_dim = {}

        if len(self.operands) + 1 != len(self.opnames):
            raise ValueError("Number of operands doesn't match the einsum string!")

        for op, op_letters in zip(self.operands, self.opnames[:-1]):
            if len(op.shape) != 0 and len(op.shape) != len(op_letters) \
                    and "..." not in op_letters and op_letters != "":
                raise ValueError("Dimension of operand " + str(op) + " doesn't match the string! " +
                                 "Shape: " + str(op.shape) + " , string: '" + op_letters + "'")

            shp = op.shape
            if op_letters[:3] == "...":
                op_letters = op_letters[::-1]
                shp = op.shape[::-1]
            for i, lett in enumerate(Einsum.split_dots(op_letters)):
                try:
                    if len(lett) == 1:
                        dim = [shp[i]]  # what if shape is an empty tuple?
                    else:
                        dim = shp[i:]
                    if self.letter_to_dim.get(lett, dim) != dim:
                        raise ValueError("Inconsistent dimension names!")
                    self.letter_to_dim[lett] = dim
                except IndexError:
                    pass  # letters that we can't add are just dimension 1

        self.shape = []
        for let in Einsum.split_dots(self.opnames[-1]):
            for l in self.letter_to_dim.get(let, [1]):
                self.shape.append(l)
        self.shape = tuple(self.shape)

    @staticmethod
    def split_dots(op_str):
        match_string = "\.{3}|\S"
        return re.findall(match_string, op_str)

    def _eval(self):
        arr = [op() for op in self.operands]

        for i, val in enumerate(arr):
            if isinstance(val, numbers.Number):
                shp = [l for let in Einsum.split_dots(self.opnames[i]) for l in self.letter_to_dim.get(let, [1])]
                arr[i] = np.broadcast_to(val, shp)

        return np.einsum(self.op_str, *arr)

    def _partial_derivative(self, wrt, previous_grad):
        """
        Usual einsum operation looks something like this c = einsum("ij,jk->ik", a, b)
        Gradient w.r.t. the first parameter just changes the op to look like this: df = einsum("ik,jk->ij", c, b).
        It basically just switches the output with one of the inputs.

        For tensors that have some of their dimensions implicitly summed, a new tensor of ones is explicitly added
        """
        order = list(range(len(self.opnames)))

        try:
            loc = self.operands.index(wrt)
        except ValueError:
            return 0
        order[loc], order[-1] = order[-1], order[loc]

        # this is concatenation of two lists in np array and then their reorder
        operands_with_grad = list(np.array(self.operands + [previous_grad])[order])

        opnames = list(np.array(self.opnames)[order])

        # here we add explicit Variables for implicitly summed out tensors
        for i, letter in enumerate(Einsum.split_dots(self.opnames[loc])):
            if letter not in Einsum.split_dots("".join(opnames[:-1])):
                opnames.insert(0, letter)

                dim = wrt.shape[i]
                var_to_insert = Variable(np.ones(dim), name="np.ones(" + str(dim) + ")")
                operands_with_grad.insert(0, var_to_insert)
        op_str = Einsum.to_einsum_string(opnames)

        return Einsum(op_str, *operands_with_grad[:-1])

    @staticmethod
    def to_einsum_string(list_of_ops):
        return ",".join(list_of_ops[:-1]) + "->" + list_of_ops[-1]



class Pow(Node):
    """
    Operation which does power of one node with other
    """
    def __init__(self, first, second, name="Pow"):
        super().__init__([first, second], name)
        self.first = self.children[0]
        self.second = self.children[1]
        self.shape = shape_from_elems(*self.children)

    def _eval(self):
        return np.power(self.first(), self.second())

    def _partial_derivative(self, wrt, previous_grad):
        if self.first == self.second == wrt:
            return previous_grad * self * (Log(self.first) + 1)
        elif self.first == wrt:
            return previous_grad * self.second * Pow(self.first, self.second - 1)
        elif self.second == wrt:
            return previous_grad * Log(self.first) * self
        return 0


class Log(Node):
    """
    Operation which takes logarithm of a node (similar to np.log())
    """
    def __init__(self, node, name="Log"):
        super().__init__([node], name)
        self.node = self.children[0]
        self.shape = self.node.shape

    def _eval(self):
        return np.log(self.node() + Node.epsilon)

    def _partial_derivative(self, wrt, previous_grad):
        if self.node == wrt:
            return previous_grad * Recipr(self.node)
        return 0


class Identity(Node):
    """
    Operation which actually does nothing to the node but forms a back-end gradient of same shape  
    """
    def __init__(self, node, name="Identity"):
        super().__init__([node], name)
        self.node = self.children[0]
        self.shape = self.node.shape

    def _eval(self):
        return self.node()

    def _partial_derivative(self, wrt, previous_grad):
        if self.node == wrt:
            return previous_grad
        return 0
class Absolute(Node):
    """
    Operation which gives absolute value of a node
    """
    def __init__(self, node, name="Absolute"):
        super().__init__([node], name)
        self.node = self.children[0]
        self.shape = self.node.shape

    def _eval(self):
        return np.abs(self.node())

    def _partial_derivative(self, wrt, previous_grad):
        if self.node == wrt:
            return previous_grad*self.node*Recipr(Pow(Pow(self.node,2),0.5))
        return 0

class Exp(Node):
    """
    Operation which gives exponential value of a node (like np.exp())
    """
    def __init__(self, node, name="Exp"):
        super().__init__([node], name)
        self.node = self.children[0]
        self.shape = self.node.shape

    def _eval(self):
        return np.exp(self.node())

    def _partial_derivative(self, wrt, previous_grad):
        if self.node == wrt:
            return previous_grad * self
        return 0
class Sine(Node):
    """
    Operation which gives sine of a node (like np.sin()) 
    """
    def __init__(self, node, name="Sine"):
        super().__init__([node], name)
        self.node = self.children[0]
        self.shape = self.node.shape

    def _eval(self):
        return np.sin(self.node())

    def _partial_derivative(self, wrt, previous_grad):
        if self.node == wrt:
            return previous_grad * Cosine(self.node)
        return 0
class Cosine(Node):
    """
    Operation which gives cosine of a node (like np.cos())
    """
    def __init__(self, node, name="Cosine"):
        super().__init__([node], name)
        self.node = self.children[0]
        self.shape = self.node.shape

    def _eval(self):
        return np.cos(self.node())

    def _partial_derivative(self, wrt, previous_grad):
        if self.node == wrt:
            return -previous_grad * Sine(self.node)
        return 0

class Tan(Node):
    
    """
    Operation which gives tan of a node (like np.tan())
    """

    def __init__(self, node, name="Tan"):
        super().__init__([node], name)
        self.node = self.children[0]
        self.shape = self.node.shape

    def _eval(self):
        return np.tan(self.node())

    def _partial_derivative(self, wrt, previous_grad):
        if self.node == wrt:
            return previous_grad * Sec(self.node)*Sec(self.node)
        return 0
class Cosec(Node):
    """
    Operation which gives cosecant of a node
    """
    def __init__(self, node, name="Cosec"):
        super().__init__([node], name)
        self.node = self.children[0]
        self.shape = self.node.shape

    def _eval(self):
        return 1/(np.sin(self.node()+Node.epsilon))

    def _partial_derivative(self, wrt, previous_grad):
        if self.node == wrt:
            return -previous_grad * Cosec(self.node)*Cot(self.node)
        return 0
class Sec(Node):
    """
    Operation which gives secant of a node
    """
    def __init__(self, node, name="Sec"):
        super().__init__([node], name)
        self.node = self.children[0]
        self.shape = self.node.shape

    def _eval(self):
        return 1/np.cos(self.node()+Node.epsilon)

    def _partial_derivative(self, wrt, previous_grad):
        if self.node == wrt:
            return previous_grad * Sec(self.node)*Tan(self.node)
        return 0
class Cot(Node):
    """
    Operation which gives cosecant of a node
    """
    def __init__(self, node, name="Cot"):
        super().__init__([node], name)
        self.node = self.children[0]
        self.shape = self.node.shape

    def _eval(self):
        return 1/np.tan(self.node()+Node.epsilon)

    def _partial_derivative(self, wrt, previous_grad):
        if self.node == wrt:
            return -previous_grad * Cosec(self.node)*Cosec(self.node)
        return 0
class Sigmoid(Node):
    """
    Operation which gives sigmoid value of a node
    """
    def __init__(self, node, name="Sigmoid"):
        super().__init__([node], name=name)
        self.node = self.children[0]
        self.shape = self.node.shape

    def _eval(self):
        return 1 / (1 + np.exp(-self.node()))

    def _partial_derivative(self, wrt, previous_grad):
        if wrt == self.node:
            return previous_grad * self * (1 - self)
        return 0
class ArcSin(Node):
    """
    Operation which gives ArcSin of a node ,whose value must lie between -1 and 1
    """
    def __init__(self, node, name="ArcSin"):
        super().__init__([node], name=name)
        self.node = self.children[0]
        self.shape = self.node.shape

    def _eval(self):
        return np.arcsin(self.node())

    def _partial_derivative(self, wrt, previous_grad):
        if wrt == self.node:
            return previous_grad * Recipr(Pow(1-Pow(self.node,2),0.5))
        return 0
class ArcCos(Node):
    """
    Operation which gives Arccos of a node,whose value must lie between -1 and 1 
    """

    def __init__(self, node, name="ArcCos"):
        super().__init__([node], name=name)
        self.node = self.children[0]
        self.shape = self.node.shape

    def _eval(self):
        return np.arccos(self.node())

    def _partial_derivative(self, wrt, previous_grad):
        if wrt == self.node:
            return -previous_grad * Recipr(Pow(1-Pow(self.node,2),0.5))
        return 0
class ArcTan(Node):
    """
    Operation which gives ArcTan value of a node, whose value can lie anywhere between -inf and +inf
    """
    def __init__(self, node, name="ArcTan"):
        super().__init__([node], name=name)
        self.node = self.children[0]
        self.shape = self.node.shape

    def _eval(self):
        return np.arctan(self.node())

    def _partial_derivative(self, wrt, previous_grad):
        if wrt == self.node:
            return previous_grad * Recipr(1+Pow(self.node,2))
        return 0
class ArcCot(Node):
    """
    Operation which gives Arccot value of a node, whose value can be anything 
    """
    def __init__(self, node, name="ArcCot"):
        super().__init__([node], name=name)
        self.node = self.children[0]
        self.shape = self.node.shape

    def _eval(self):
        return np.arctan(1/(self.node()+Node.epsilon))

    def _partial_derivative(self, wrt, previous_grad):
        if wrt == self.node:
            return -previous_grad * Recipr(1+Pow(self.node,2))
        return 0

class ArcSec(Node):
    """
    Operation which gives Arcsec value of a node , whose value can be anything except between -1 and 1
    """
    def __init__(self, node, name="ArcSec"):
        super().__init__([node], name=name)
        self.node = self.children[0]
        self.shape = self.node.shape

    def _eval(self):
        return np.arccos(1/(self.node()+Node.epsilon))

    def _partial_derivative(self, wrt, previous_grad):
        if wrt == self.node:
            return previous_grad * Recipr(Absolute(self.node)*Pow(Pow(self.node,2)-1,0.5))
        return 0
class ArcCosec(Node):
    """
    Operation which gives Arccosec value of a node, whose value can be anything except between -1 and 1
    """
    def __init__(self, node, name="ArcCosec"):
        super().__init__([node], name=name)
        self.node = self.children[0]
        self.shape = self.node.shape

    def _eval(self):
        return np.arcsin(1/(self.node()+Node.epsilon))

    def _partial_derivative(self, wrt, previous_grad):
        if wrt == self.node:
            return -previous_grad * Recipr(self.node*Pow(Pow(self.node,2)-1,0.5))
        return 0



"""
Below are module wrappers defining all the hyperbolic trigonometric functions
"""
@module_wrapper
def Tanh(x):
    val = Exp(-2 * x)
    return (1 - val) / (1 + val)

@module_wrapper
def Sinhx(x):
    val = Exp(x) - Exp(-x)
    return val/2
@module_wrapper
def Coshx(x):
    val = Exp(x) + Exp(-x)
    return val/2
@module_wrapper
def Sechx(x):
    val = Exp(x) + Exp(-x)
    return 2/val
@module_wrapper
def Cosech(x):
    val = Exp(x) - Exp(-x)
    return 2/val
@module_wrapper
def Coth(x):
    val = Exp(-2 * x)
    return (1 + val) / (1 - val)

@module_wrapper
def SquaredDifference(x, y):
    diff = x - y
    return diff * diff


"""
Below are some important matrix and tensor operations defined as module wrappers around Einsum
Not a complete list but enncompasses the important ones such as: Matrix multplication, transpose, Inner,outer and hadamard products
"""

@module_wrapper
def MatMulV(x,y):
    return Einsum("j,ij->j",x,y)

@module_wrapper
def MatMul(x, y):
    return Einsum("ij,jk->ik", x, y)


@module_wrapper
def Transpose(x):
    return Einsum("ij->ji", x)



@module_wrapper
def Sum1DArray(x):
    return Einsum("i->",x)
@module_wrapper
def ElementwiseMul1D(x,y):
    return Einsum("i,i->i",x,y)
@module_wrapper
def InnerProduct1D(x,y):
    return Einsum("i,i->",x,y)
@module_wrapper
def OuterProduct1D(x,y):
    return Einsum("i,j->ij",x,y)
@module_wrapper
def Trace(x):
    return Einsum("ii->",x)
@module_wrapper
def Diag(x):
    return Einsum("ii->i",x)
@module_wrapper
def Hadamard(x,y):
    return Einsum("ij,ij->ij",x,y)



