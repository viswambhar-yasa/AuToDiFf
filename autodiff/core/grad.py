"""
Author: Viswambhar Yasa
Grad function to collect all the partial derivatives and get the gradient


CITATION: This function is directly taken from "https://github.com/bgavran/autodiff"
"""



#Import required packages
import functools
import numpy as np
import collections
from .utils import reverse_topo_sort
from .ops import Add
from .node import Variable


def grad(top_node, wrt_list, previous_grad=None):
    """
    Transforms the computational graph of top_node into a list of computational graphs corresponoding to
    partial derivatives of top_node with respect to all variables in wrt_list.

    It delegates the actual implementation of partial derivatives to nodes in the computational graph and doesn't care
    how they're implemented.
    It can be elegantly implemented using foldl.
    Essentially, grad is structural transformation that is a function *only* of the topology of the computational graph.

    :param top_node: node in the graph whose gradient will be taken with respect to all variables in wrt_list
    :param wrt_list: list of objects, instances of Node, whose gradient we're looking for
    :param previous_grad: incoming gradient to top node, by default np.ones(top_node.shape)
    :return: returns a list of gradients corresponding to variables in wrt_list
    """
    assert isinstance(wrt_list, list) or isinstance(wrt_list, tuple)
    if previous_grad is None:
        previous_grad = Variable(np.ones(top_node.shape), name=add_sum_name(top_node))

    dct = collections.defaultdict(lambda: Variable(0))
    dct[top_node] += previous_grad  # add the incoming gradient for the top node

    def add_partials(dct, node):
        for child in set(node.children):  # calc. all partial derivs w.r.t. each child and add them to child's grads
            dct[child] += node.partial_derivative(wrt=child, previous_grad=dct[node])
        return dct

    dct = functools.reduce(add_partials, reverse_topo_sort(top_node), dct)  # basically a foldl

    return [dct[wrt] for wrt in wrt_list]


def add_sum_name(node):
    return "'" + node.name + "' grad_sum"
