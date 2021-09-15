"""
Author: Viswambhar Yasa

The function to wrap something which is not instantiated in Autodiff framework but comes across during operations.

CITATION: This function is directly taken from "https://github.com/bgavran/autodiff"
"""


#Import required packages
from .node import Node
from .grad import grad


def checkpoint(fn):
    """
    The checkpoint function
    inputs:
    fn: function to be wrapped
    returns wrap in primitice function of that specific function

    """

    def wrap_in_primitive(*fn_args):
        """
        The Wrap in primitive function to convert the undefined function compatible with Autodiff 
        """
        op = Node(children=fn_args, name=fn.__name__)

        op._eval = lambda: fn(*fn_args)()
        op._partial_derivative = lambda wrt, previous_grad: grad(fn(*fn_args), [wrt], previous_grad=previous_grad)[0]
        # should graph_df return the already called Node() or just the Node, like right now?

        return op

    return wrap_in_primitive


