"""
Shape manipulators of arrays within the framework.
CITATION: These functions are based(but not directly taken from) on reshape implementations "https://github.com/bgavran/autodiff" . Slice was taken directly from there.
"""

#Import required packages
import numbers
import numpy as np
from .node import Node





class Concat(Node):
    """
    Class to concatenate two arrays
    """
    def __init__(self, a, b, axis=0):
        """
        params:
        a,b: input arrays to be concatenated
        axis: Axis along which concatenation should take place 
        """
        #Sanctity check
        assert axis >= 0  
        super().__init__([a, b], name="Concat")
        self.a, self.b = self.children
        self.shape = list(self.a.shape)
        self.axis = axis
        self.shape[axis] += self.b.shape[axis]

    def _eval(self):
        a_val = self.a()
        b_val = self.b()
        return np.concatenate((a_val, b_val), axis=self.axis)

    def _partial_derivative(self, wrt, previous_grad):
        previous_grad = Reshape(previous_grad, self.shape)  
        split = self.a.shape[self.axis]

        slice_val = [slice(None, None, None) for _ in range(self.axis + 1)]
        if wrt == self.a:
            slice_val[self.axis] = slice(None, split, None)
            return previous_grad[slice_val]
        elif wrt == self.b:
        
            slice_val[self.axis] = slice(split, None, None)
            return previous_grad[slice_val]
        return 0








class Slice(Node):
    def __init__(self, node, slice_val, name="Slice"):
        if name is None:
            name = str(slice_val)
        super().__init__([node], name)
        self.node = self.children[0]
        self.slice_val = slice_val

        self.shape = np.zeros(self.node.shape)[self.slice_val].shape

    def _eval(self):
        val = self.node()
        return val[self.slice_val]

    def _partial_derivative(self, wrt, previous_grad):
        
        if self.node == wrt:
            grad = np.zeros(wrt.shape)
            grad[self.slice_val] = previous_grad()
            return grad
        return 0


class Pad(Node):
    def __init__(self, node, pad_width, constant_values, name="Slice"):
        """
        :param node:
        :param pad_width:  different than pad_width arg in np.pad, this one pads up to the length provided
        :param constant_values:
        :param name:
        """
        super().__init__([node], name)
        self.node = self.children[0]
        self.pad_width = pad_width
        self.constant_values = constant_values

        self.shape = np.pad(np.ones(self.node.shape),
                            self.pad_width,
                            mode="constant",
                            constant_values=self.constant_values).shape

    def _eval(self):
        val = self.node()
        return np.pad(val,
                      self.pad_width,
                      mode="constant",
                      constant_values=self.constant_values)

    def _partial_derivative(self, wrt, previous_grad):
        if self.node == wrt:
            slice_val = [slice(pad[0], shp - pad[1]) for pad,shp in zip(self.pad_width, self.shape)]
            return previous_grad[slice_val]
        return 0
class ReduceSumKeepDims(Node):
    """
    Operation which reduces the array keeping the dimensions by adding accordingly
    """
    def __init__(self, node, axes):
        """
        params:
        node: The array which needs to be reduced
        axes: Axes which need to be kept
        """
        super().__init__([node], name="ReduceSumKeepDims")
        self.axes = tuple(axes)
        self.node = self.children[0]
        self.shape = [1 if i in self.axes else shp for i, shp in enumerate(self.node.shape)]

    def _eval(self):
        return np.sum(self.node(), axis=self.axes, keepdims=True)

    def _partial_derivative(self, wrt, previous_grad):
        if self.node == wrt:
            return previous_grad * np.ones(self.node.shape)
        return 0


class Reshape(Node):
    """
    Operation to reshape a given array
    """
    def __init__(self, node, shape, name="Reshape"):
        """
        params:
        node:the array to be reshaped 
        shape : the shape to be reshaped into 
        """
        super().__init__([node], name)
        
        self.shape = self.infer_shape(shape) 
        self.node = self.children[0]
    def infer_shape(self, shape):
        """
        Special method to infer shape 
        params: 
        shape: the output shape 
        returns the shape 
        
        """
        if isinstance(shape, numbers.Number):
            return shape
        if -1 in shape:
            shape = list(shape)
            for i in range(len(shape)):
                if shape[i] == -1:
                    shape[i] = int(-np.prod(self.node.shape) / np.prod(shape))
        return shape  

    def _eval(self):
        node_val = self.node()
        if isinstance(node_val, numbers.Number):
            return np.broadcast_to(node_val, self.shape)
        return np.reshape(node_val, self.shape)

    def _partial_derivative(self, wrt, previous_grad):
        if self.node == wrt:
            return Reshape(previous_grad, self.node.shape)
        return 0