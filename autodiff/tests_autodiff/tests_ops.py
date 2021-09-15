"""
Author: Viswambhar Yasa

This file contains all the  test cases checking the functionality of all the operations defined in the ops.py file
All the tests can be run by command "pytest tests_ops.py"

"""


#Import required packages
import pytest
import numpy as np 
from autodiff.core.ops import *
from autodiff.core.node import *
import autodiff as ad
from autodiff.core.grad import grad



def test_ReduceSumToShape1_as_scalar():
    """
    Aim: Test whether ReduceSumToShape(which is a wrapper of ReduceSumKeepDims) exactly reduces an array to a scalar
    Expected Result: 25
    Obtained Result: 25
    Remarks:None
    """
    X= np.ones((5,5))
    X = Variable(X,"X")
    val = ReduceSumToShape(X,())
    assert val() == 25 and isinstance(val,ReduceSumKeepDims)

def test_ReduceSumToShape2_as_column():
    """
    Aim: Test whether ReduceSumToShape(which is a wrapper of ReduceSumKeepDims) exactly reduces an array to a column vector
    Expected Result: [5,5,5,5,5]
    Obtained Result: [5,5,5,5,5]
    Remarks:None

    """
    X = np.ones((5,5))
    X = Variable(X,"X")
    val = ReduceSumToShape(X,(5,))
    print(val())
    flag = np.array_equal(val(),[5., 5., 5., 5., 5.])
    assert flag == True and isinstance(val,ReduceSumKeepDims)


def test_ReduceSumToShape3_as_row():
    """
    Aim: Test whether ReduceSumToShape(which is a wrapper of ReduceSumKeepDims) exactly reduces an array into a single row
    Expected Result: [[5,5,5,5,5]]
    Obtained Result: [[5,5,5,5,5]]
    Remarks:None
    """
    X = np.ones((5,5))
    X = Variable(X,"X")
    val = ReduceSumToShape(X,(1,5))
    print(val())
    flag = np.array_equal(val(),np.full((1,5),5.))
    assert flag == True and isinstance(val,ReduceSumKeepDims)

def test_add1_2equal():
    """
    Aim:Test addition operation (if the value is correct and derivatives are passed as expected) 
    Expected Result:Add: [[2,4],[8,10]] dzdx=dzdy=[[1,1],[1,1]]
    Obtained Result:[[2,4],[8,10]]
    Remark: For the case where two variables are of same shape
    """
    x = np.array([[1,2],[4,5]])
    y = np.array([[1,2],[4,5]])
    X = Variable(x,"X")
    Y = Variable(y,"Y")
    Z = x+y

    Z1 = Add(X,Y)
    dzdx,dzdy = grad(Z1,[X,Y])
    assert np.array_equal(Z,Z1()) == True and isinstance(Z1,Add) and np.array_equal(dzdx(),np.ones_like(x)) and np.array_equal(dzdy(),np.ones_like(x))


def test_add2_matrix_vector():
    """
    Aim:Test addition operation (if the value is correct and derivatives are passed as expected) 
    Expected Result:add = [[2,3][5,6]] and dzdx=[2,2],dzdy=[[1,1],[1,1]]
    Obtained Result:
    Remarks: For the case where one variable is matrix and other is 1D-vector

    """
    x = np.ones((2,))
    y = np.array([[1,2],[4,5]])
    Z = x+y
    X = Variable(x,"X")
    Y = Variable(y,"Y")

    
    Z1 = Add(X,Y)
    dzdx,dzdy = grad(Z1,[X,Y])
    print(dzdx())
    print(np.full_like(x,1.0))
    assert np.array_equal(Z,Z1()) == True and isinstance(Z1,Add) and np.array_equal(dzdx(),np.full_like(x,2)) and np.array_equal(dzdy(),np.full_like(y,1.0,dtype=type(dzdy())))



def test_add3_matrix_vector_fractions():
    """
    Aim:Test addition operation (if the value is correct and derivatives are passed as expected) 
    Expected:[ 2.,  3.],
             [-3.,  1.00324465]] and dzdx=[[2,2]] dzdy=[[1,1],[1,1]]
    Obtained:[ 2.,  3.],
             [-3.,  1.00324465]] and dzdx=[[2,2]] dzdy=[[1,1],[1,1]]
    Remarks: For a 2D array and another 2D array as a row vector
    """
    x = (np.ones((1,2)))
    y = np.array([[1,2.0],[-4,5/1541]])
    Z = x+y
    X =Variable(x,"X")
    Y = Variable(y,"Y")
    Z1 = Add(X,Y)
    
    dzdx,dzdy = grad(Z1,[X,Y])
    print(dzdx())
    assert np.array_equal(Z,Z1()) == True and isinstance(Z1,Add) and np.array_equal(dzdx(),np.full_like(x,2)) and np.array_equal(dzdy(),np.ones_like(y) )



def test_add4_irrational():
    """
    Aim:Test addition operation (if the value is correct and derivatives are passed as expected) 
    Expected:add=13.141592653589793 dzdx=dzdy=1
    Obtained:add=13.141592653589793 dzdx=dzdy=1
    Remark: This test case ascertains the behaviour of Add when irrational numbers are passed and derivatives are also passed as expected 

    """
    x = np.pi
    y = 10
    Z = x+y
    X = Variable(x,"X")
    Y = Variable(y,"Y")
    Z1 = Add(X,Y)
    dzdx,dzdy = grad(Z1,[X,Y])

    assert np.array_equal(Z,Z1()) == True and isinstance(Z1,Add) and dzdx()==dzdy()==1 
"""
def test_add5_complex_meant_to_fail():
    X = np.pi
    print("This is meant to fail to show that Complex numbers are not supported")
    Y = Variable(10j)
    Z = X+Y
    #Z1 = Add(X,Y)
    
    assert  isinstance(Y,TypeError)
"""
def test_Mul1_irrational():
    """
    Aim:Test the Mul operation (if the value is correct and derivatives are passed as expected) 
    Expected:pi ,dzdx=pi,dzdy=1
    Obtained:pi,dzdx=pi,dzdy=1
    Remark:multiplying an irrational integer to check robustness of kernel 
    """
    x = 1
    y = np.pi
    Z = x*y
    X = Variable(x,"X")
    Y = Variable(y,"Y")
    Z1 = Mul(X,Y)
    dzdx,dzdy = grad(Z1,[X,Y])
    assert isinstance(Z1,Mul) and Z1()==np.pi and dzdx()==y and dzdy()==x

def test_Mul2_matrix_scalar():
    """
    Aim:Test the Mul operation (if the value is correct and derivatives are passed as expected) 
    Obtained: [[3 6],[6 6]] and dzdx=[[3 3] [3 3]] , dzdy=7 
    Expected: [[3 6],[6 6]] and dzdx=[[3 3] [3 3]] , dzdy=7 
    Remarks: Multiplying a scalar with an array to check that derivatives are broadcasted correctly
    """
    x = np.array([[1,2],[2,2]])
    y = 3
    
    Z = x*y
    X = Variable(x,"X")
    Y= Variable(y,"Y")
    Z1 = Mul(X,Y)
    dzdx,dzdy = grad(Z1,[X,Y])
    print(dzdx())
    print(dzdy())

    assert isinstance(Z1,Mul) and np.array_equal(Z1(),Z)==True and np.array_equal(dzdx(),np.full_like(x,y)) and np.array_equal(dzdy(),7)


def test_Mul3_matrix_vector():
    """
    Aim:Test the Mul operation (if the value is correct and derivatives are passed as expected)
    Obtained: mul=[[1 0][0 0]]    and dzdy=[[1 0] [1 0]]dzdx=[1,0.423310825130748]
    Expected: mul=[[1 0][0 0]]    and dzdy=[[1 0] [1 0]]dzdx=[1,0.423310825130748]


    Remarks: Multiplying a scalar with a vector to check that derivatives are broadcasted correctly


    """
    x = np.array([1,0])
    y = np.array([[1,np.pi],[0,-np.exp(1)]])
    Z = x*y
    X = Variable(x,"X")
    Y = Variable(y,"Y")
    Z1 = Mul(X,Y)
    dzdx,dzdy = grad(Z1,[X,Y])
    print(dzdx())
    print(dzdy())
    assert isinstance(Z1,Mul) and np.array_equal(Z1(),Z)==True and np.array_equal(dzdy(),np.array([[1.,0.],[1.,0.]])) and np.array_equal(dzdx(),np.array([1,np.pi-np.exp(1)]))

def test_Negate_zero():
    """
    Aim: To check the negation operation((if the value is correct and derivatives are passed as expected))
    Expected: 0, dydx = -1
    Obtained:0, dydx = -1
    Remarks:This is a typical case , because the negation of zero is itself but the gradient passed should change it's sign.
    """
    X = Variable(0,"X")
    Y = Negate(X)
    dy = grad(Y,[X])[0]
    assert isinstance(Y,Negate) and Y()==0 and dy()==-1

def test_Negate_array():
    """
    Aim: To check the negation operation((if the value is correct and derivatives are passed as expected))
    Expected: [[-np.pi,-2.0],[+3,-4.2222222]], dydx = [[-1 -1][-1 -1]]
    Obtained:[[-np.pi,-2.0],[+3,-4.2222222]], dydx =  [[-1 -1][-1 -1]]
    Remarks: taking all possible types of numbers and negating them
    """
    x = np.array([[np.pi,2.0],[-3,4.2222222]])
    X = Variable(x,"X")
    Y = Negate(X)
    dy = grad(Y,[X])[0]


    assert isinstance(Y,Negate) and np.array_equal(Y(),-1*x) == True and np.array_equal(dy(),np.full((2,2),-1.)) 

def test_recipr_scalar():
    """
    Aim: To test the operation Reciprocal(which basically does element-wise reciprocal) 
    Obtained: 0.19999999999996 and dy = -0.04
    Expected: 0.2 and dy = -0.04
    Remarks: This small deviation marks the distinctive feature employed in all the operations which might have to deal with unwanted infinities. In this case division by zero is avoided by adding a very small innate error of 1e-12 
    """
    x= 5
    X = Variable(x,"X")
    Y= Recipr(X)
    dy = grad(Y,[X])[0]



    assert isinstance(Y,Recipr) and np.abs(Y() - (1/(5+1e-12))) < 0.0000001 and np.abs(dy()  - np.array(-0.04)) < 0.000000001

def test_recipr_irrational_array():
    """
    Aim:To test the operation Reciprocal(which basically does element-wise reciprocal) 
    Expected:[[0.31830989, 0.31818182], dy = [[1.01321184e-01, 1.01239669e-01],
       [2.71828183, 0.082085  ]]                [7.38905610e+00, 6.73794700e-03]]
    Obtained:[[0.31830989, 0.31818182],    dy= [[1.01321184e-01, 1.01239669e-01],
              [2.71828183, 0.082085  ]]        [7.38905610e+00, 6.73794700e-03]]
    Remarks:Deviation due to buffer avoiding infinities and invoking itself  for passage of derivatives is also tested. 
    """
    x = np.array([[np.pi,22/7],[np.exp(-1),np.exp(2.5)]])
    X = Variable(x,"X")
    Y = Recipr(X)
    
    dy = grad(Y,[X])[0]
    print(dy())
    print(-np.reciprocal((x+1e-12)*(x+1e-12)))
    assert isinstance(Y,Recipr) and np.array_equal(Y(),np.reciprocal(x+1e-12)) and np.all(np.less(np.abs(dy()+np.reciprocal((x+1e-12)*(x+1e-12))) , np.full_like(dy(),0.0000001)))

def test_einsum_onearg_identity():
    """
    Aim:To test the operation Einsum
    Expected: random array of shape (2,3,5,7)
    Obtained: the same random array of shape (2,3,5,7)
    Remarks: Same indices on both sides of arrow of an einsum means Identity wrapper of einsum is tested here
    """
    X = np.random.randn(2,3,5,7)
    Xv = Variable(X,"Xv")
    Z = Einsum("ijkl->ijkl",Xv)
    assert isinstance(Z,Einsum) and np.array_equal(Z(),X) == True

def test_einsum_onearg_sum_1axis():
    """
    Aim:To test the operation Einsum
    Expected: contracted random array(4th order tensor ) along the last dimension
    Obtained: contracted random array(4th order tensor ) along the last dimension
    Remarks: this test  tests the single contraction of a 4th order tensor with itself(i.e reduce sum keep dimensions 1,2,3) 

    """
    X = np.random.randn(2,3,5,7)
    Xv = Variable(X,"Xv")
    Z = Einsum("ijkl->ijk",Xv)
    assert isinstance(Z,Einsum) and np.array_equal(Z(),np.einsum("ijkl->ijk",X)) == True

def test_einsum_onearg_sum_2axis():
    """
    Aim:To test the operation Einsum 
    Expected: contracted array along the last two dimensions
    Obtained: contracted array along the last two dimensions
    Remarks: this test tests double contraction of a 4th order tensor with itself(i.e reduce sum keep dimensions 1,2) 
    """
    X = np.random.randn(2,3,5,7)
    Xv = Variable(X,"Xv")
    Z = Einsum("ijkl->ij",Xv)
    assert isinstance(Z,Einsum) and np.array_equal(Z(),np.einsum("ijkl->ij",X)) == True


def test_einsum_onearg_sum_3axis():
    """
    Aim:To test the operation Einsum 
    Expected: contracted array along the last three dimensions
    Obtained: contracted array along the last three dimensions
    Remarks: this test tests the triple contraction of a 4th order tensor with itself(i.e reduce sum keep dimensions 1) 
    """
    X = np.random.randn(2,3,5,7)
    Xv = Variable(X,"Xv")
    Z = Einsum("ijkl->i",Xv)
    assert isinstance(Z,Einsum) and np.array_equal(Z(),np.einsum("ijkl->i",X)) == True

def test_einsum_onearg_sum_allaxis():
    """
    Aim:To test the operation Einsum 
    Expected:contracted array along the last all dimensions
    Obtained:contracted array along the last all dimensions
    Remarks: this tests einsum as reduce sum keep dims none 

    """
    X = np.random.randn(2,3,5,7)
    Xv = Variable(X,"Xv")
    Z = Einsum("ijkl->",Xv)
    assert isinstance(Z,Einsum) and np.array_equal(Z(),np.einsum("ijkl->",X)) == True

def test_einsum_matmul():
    """
    Aim:To test the operation Einsum (Value and its derivatives)
    Expected: z=[[ 433,  344], dzdx = [[115, 108] dzdy =[[55, 55],
               [3452, 3176]]         [115, 108]]       [10, 10]]
    obtained:z=[[ 433,  344],  dzdx = [[115, 108], dzdy=[[55, 55],
              [3452, 3176]]          [115, 108]]        [10, 10]]
    Remarks: This tests usage of Einsum for matrix multiplication(tests the wrapper function Wrapper Matmul)
    """

    x = np.array([[3,4],[52,6]])
    y = np.array([[59,56],[64,44]])
    z = np.dot(x,y)
    X = Variable(x,"X")
    Y = Variable(y,"Y")
    Z = Einsum("ij,jk->ik",X,Y)
    dzdx , dzdy = grad(Z,[X,Y])
    assert isinstance(Z,Einsum) and np.array_equal(Z(),z) and np.array_equal(dzdx(),np.einsum("ik,jk->ij",np.ones_like(z),y)) \
        and np.array_equal(dzdy(),np.einsum("ij,jk->jk",x,np.ones_like(z)))

def test_einsum_matmul3():
    """
    Aim:To test the operation Einsum (Value and its derivatives)
    Expected: z=[[ 51503,  43169], dzdx =[[20928,  7972], dzdy=[[11000,  8965]  dzdw =[[3285, 3285],
                 [673332, 462756]]        [20928,  7972]]       [ 2000,  1630]]        [3520, 3520]]
    Obtained: z=[[ 51503,  43169],  dzdx =[[20928,  7972], dzdy=[[11000,  8965]  dzdw= [[3285, 3285],
                [673332, 462756]]          [20928,  7972]]       [ 2000,  1630]]         [3520, 3520]]
    Remarks: This test tests usage of einsum for multplication of 3 matrices 
    """
    x = np.array([[3,4],[52,6]])
    y = np.array([[59,56],[4,44]])
    w = np.array([[151,49],[65,98]])
    z = np.dot(np.dot(x,y),w)
    X = Variable(x,"X")
    Y = Variable(y,"Y")
    W = Variable(w,"W")
    Z = Einsum("ij,jk,kl->il",X,Y,W)
    dzdx,dzdy,dzdw = grad(Z,[X,Y,W]) 
    print(dzdx())
    print(np.einsum("il,jk,kl->ij",np.ones_like(z),y,w))
    assert isinstance(Z,Einsum) and np.array_equal(Z(),z) and np.array_equal(dzdx(),np.einsum("il,jk,kl->ij",np.ones_like(z),y,w)) \
        and np.array_equal(dzdy(),np.einsum("ij,il,kl->jk",x,np.ones_like(z),w)) and np.array_equal(dzdw(),np.einsum("ij,jk,il->kl",x,y,np.ones_like(z)))


def test_einsum_different_indices():
    """
    Aim:To test the operation Einsum (Value and its derivatives)
    Expected: third order tensor and second order derivatives obtained by outer product of three second order tensors

    Obtained: third order tensor and second order derivatives obtained by outer product of three second order tensors
    
    Remarks: This test tests usage of einsum for outer product of three second order tensors to get third order tensor

    """
    x = np.array([[3,4],[52,6]])
    y = np.array([[59,54],[44,84]])
    w = np.array([[11,29],[75,9]])
    z = np.einsum("ij,jk,kl->ijl",x,y,w)
    X = Variable(x,"X")
    Y = Variable(y,"Y")
    W = Variable(w,"W")
    Z = Einsum("ij,jk,kl->ijl",X,Y,W)
    dzdx,dzdy,dzdw = grad(Z,[X,Y,W])
    assert isinstance(Z,Einsum) and np.array_equal(Z(),z) and np.array_equal(dzdx(),np.einsum("ijl,jk,kl->ij",np.ones_like(z),y,w)) \
        and np.array_equal(dzdy(),np.einsum("ij,ijl,kl->jk",x,np.ones_like(z),w)) and np.array_equal(dzdw(),np.einsum("ij,jk,ijl->kl",x,y,np.ones_like(z)))

def test_einsum_4thorder():
    """
    Aim:To test the operation Einsum (Value and its derivatives)
    Expected: fourth order tensor and second order derivatives obtained by outer product of three second order tensors
    Obtained: fourth order tensor and second order derivatives obtained by outer product of three second order tensors
    Remarks: This test tests usage of einsum for outer product of three second order tensors to get fourth order tensor


    """
    x = np.array([[35,24],[52,6]])
    y = np.array([[59,56],[44,44]])
    w = np.array([[11,45],[75,28]])
    z = np.einsum("ij,jk,kl->ijkl",x,y,w)
    X = Variable(x,"X")
    Y = Variable(y,"Y")
    W = Variable(w,"W")
    Z = Einsum("ij,jk,kl->ijkl",X,Y,W)
    dzdx,dzdy,dzdw = grad(Z,[X,Y,W])
    assert isinstance(Z,Einsum) and np.array_equal(Z(),z) and np.array_equal(dzdx(),np.einsum("ijkl,jk,kl->ij",np.ones_like(z),y,w)) \
        and np.array_equal(dzdy(),np.einsum("ij,ijkl,kl->jk",x,np.ones_like(z),w)) and np.array_equal(dzdw(),np.einsum("ij,jk,ijkl->kl",x,y,np.ones_like(z)))


def test_pow_scalar():
    """
    Aim: Test the operation power(value and derivatives)
    Expected: val = 9 and dy =6
    Obtained: val = 9 and dy =6
    Remarks: tests specific x**n situation n being a scalar whose derivative will be n*x**n-1
    """
    x = 3
    y = 3**2
    X = Variable(x,"X")
    Y = Pow(X,2)
    dy = grad(Y,[X])[0]

    assert isinstance(Y,Pow) and Y()==9 and dy()==6

def test_pow_scalar_irrational():
    """
    Aim: Test the operation power(value and derivatives)
    Obtained:val= 0.03170146783514191 ,dy = -0.03319769948629831
    Expected:val = 0.03170146783514191, dy = -0.03319769948629831
    Remarks: tests specific x**n stuation n being an irrational number.


    """
    x = 3
    y = 3**-np.pi
    X = Variable(x,"X")
    Y = Pow(X,-np.pi)
    dy = grad(Y,[X])[0]


    assert isinstance(Y,Pow) and Y()==y and dy()==-np.pi*3**(-np.pi-1)
def test_pow_array_with_scalar():
    """
    Aim: Test the operation power(value and derivatives)
    obtained: value is element wise squared of a random array,derivative is 2*random array 
    Expected: value is element wise squared of a random array,derivative is 2*random array 
    Remarks: tests how pow operation for a higher dimensional array
    """
    x = np.random.rand(3,3,3)
    y = x**2
    X = Variable(x,"X")
    Y = Pow(X,2)
    dy = grad(Y,[X])[0]


    assert isinstance(Y,Pow) and np.array_equal(Y(),y) and np.array_equal(dy(),2*x)

def test_pow_array_with_itself():
    """
    Aim: Test the operation power(value and derivatives)
    Expected: Each value in third order tesnor raised to itself , derivative is additionally multiplied with log of value 
    Obtained:Each value in third order tesnor raised to itself, derivative is very slightly less than expected 
    Remarks: tests how pow operation works for type x**x whose derivative is x**x*log(x) and the slight deviation is result of avoiding the unwanted infinities in log .

    """
    x = np.random.rand(3,3,3)
    y = x**x
    
    X = Variable(x,"X")
    Y = Pow(X,X)
    #print(Y())
    dy = grad(Y,[X])[0]
    print(dy())
    print((x**x)*(np.log(x)+1))


    assert isinstance(Y,Pow) and np.array_equal(Y(),y) and np.array_equal(dy(),(x**x)*(np.log(x+1e-12)+1))

def test_pow_array_with_another():
    """
    Aim:Test the operation power(value and derivatives)
    Expected:Each value in third order tesnor raised to another value of another tensor with same index , derivative is additionally multiplied with log of value 
    Obtained:Each value in third order tesnor raised to another value of another tensor with same index , derivative is additionally multiplied with log of value , small deviation due to avoiding infinities.
    Remarks: tests both scenarios x**a and a**x and derivatives
    """
    x = np.random.rand(4,4,4)
    z = np.random.rand(4,4,4)
    X = Variable(x,"X")
    Z = Variable(z,"Z")
    y = x**z
    print(y)
    Y = Pow(X,Z)
    print(Y())
    dydx ,dydz= grad(Y,[X,Z])



    assert isinstance(Y,Pow) and np.array_equal(Y(),y) and np.array_equal(dydx(),z*(x**(z-1))) and np.array_equal(dydz(),(x**z)*np.log(x+1e-12))


def test_log_scalar():
    """
    Aim: test the log operation(value and derivative)
    Expected: 1.6094379124341003 and dy = 0.2 
    Obtained: 1.6094379124343003 and dy = 0.20000000000100002
    Remarks: The slight deviation is caused by a very small value added during the operation to avoid infinities
    """
    x = 5 
    y = np.log(x+1e-12)
    X = Variable(x,"X")
    Y = Log(X)
    dy = grad(Y,[X])[0]
    assert isinstance(Y,Log) and Y()==y and dy()==1/(5+1e-12)

def test_log_0():
    """
    Aim: test the log operation(value and derivative)
    Expected: Not defined/Does not exist
    Obtained: -27.631021115928547 and dy = 10000000000000
    Remarks: This shows the anomalous behaviour at discontinuities , they are avoided by adding a very small number.This is employed because they are more likely to occur in NN and discontinuites need to be dealt with.
    """
    x = 0 
    y = np.log(x+1e-12)
    X = Variable(x,"X")
    Y = Log(X)
    dy = grad(Y,[X])[0]
    assert isinstance(Y,Log) and Y()==y and dy()==1/1e-12
def test_log_array():
    """
    Aim: Test the log operation(value and derivative)
    Expected: np.log of the array
    Obtained: np.log(array+1e-12)
    Remarks: The small error constitutes weight to avoid infinities
    """
    x = np.array([[np.pi,np.exp(1)],[232,848]])
    y = np.log(x+1e-12)
    X = Variable(x,"X")
    Y = Log(X)
    dy=grad(Y,[X])[0]
    print(dy())
    print(1/x)

    assert isinstance(Y,Log) and np.array_equal(Y(),y) and np.array_equal(dy(),1/(x+1e-12))

def test_identity():
    """
    Aim: Test the identity function (function and derivative)
    Expected: The same function which is passed as argument
    Obtained:The same function which is passed as argument
    Remarks: Sometimes derivatives musr also pass without any changes , This Identity is used in such conditions
    """
    x = np.array([[np.pi,np.exp(1)],[232.3864641,-84.448]])
    X = Variable(x,"X")
    y = Identity(X)
    dy=grad(y,[X])[0]
    print(dy())
    print(y())
    
    assert isinstance(y,Identity) and np.array_equal(x,y()) and np.array_equal(dy(),np.full_like(x,1.))

def test_exp_scalar():
    """
    Aim: Test the exponent function (value and derivative)
    Expected: exp(5) 
    Obtained:exp(5)
    Remarks: The derivative of exp(x) is again exp(x)

    """
    x = 5 
    y = np.exp(x)
    X = Variable(x,"X")
    Y = Exp(X)
    dy=grad(Y,[X])[0]
    assert isinstance(Y,Exp) and Y()==y and dy()==y
def test_exp_array():
    """
    Aim: Test the exponent function(value and derivative)
    Expected: exp(array)
    Obtained: exp(array)
    Remarks: The derivative is equal to value 
    """
    x = np.random.rand(2,2)
    y = np.exp(x)
    X = Variable(x,"X")
    Y = Exp(X)
    dy=grad(Y,[X])[0]
    assert isinstance(Y,Exp) and np.array_equal(Y(),y) and np.array_equal(dy(),y)


def test_sine_scalar():
    """
    Aim: test the sine function(value and derivative)
    Expected:1
    Obtained:1
    Remarks: Exact derivatives are obtained because of no presence of discontinuities
    """
    x = np.pi/2
    y = np.sin(x)
    
    X = Variable(x,"X")
    Y = ad.Sine(X)
    dy = grad(Y,[X])[0]
    print(dy())
    assert isinstance(Y,Sine) and y==Y() and dy()==np.cos(x)

def test_sine_array():
    """
    Aim:test the sine function (value and derivative)
    Expected:sine(array)
    Obtained:sine(array)
    Remarks:Exact derivatives are obtained because of no presence of discontinuities
    """
    x = np.random.rand(4,5,6)
    y = np.sin(x)
    X = Variable(x,"X")
    Y = ad.Sine(X)
    dy = grad(Y,[X])[0]
    print(dy())
    assert isinstance(Y,Sine) and np.array_equal(Y(),y) and np.array_equal(dy(),np.cos(x))

def test_cosine_array():
    """
    Aim: test cosine function (value and derivative)
    Expected:cosine(array)
    Obtained:cosine(array)
    Remarks:Exact derivatives are obtained because of no presence of discontinuities
    """
    x = np.random.rand(4,5)
    y = np.cos(x)
    X = Variable(x,"X")
    Y = ad.Cosine(X)
    
    dy = grad(Y,[X])[0]
    print(dy())
    print(-np.sin(x))
    assert isinstance(Y,Cosine) and np.array_equal(Y(),y) and np.array_equal(dy(),-np.sin(x))

def test_cosine_scalar():
    """
    Aim:test cosine function (value and derivative)
    Expected:1
    Obtained:1
    Remarks:Exact derivatives are obtained because of no presence of discontinuities
    """
    x = 0
    y = np.cos(x)
    Y = Cosine(x)
    dy = grad(Y,[x])[0]
    print(dy())
    print(-np.sin(x))
    assert isinstance(Y,Cosine) and y==Y() and dy()==-np.sin(x)

def test_tan_scalar():
    """
    Aim:test the tan function (value and derivative)
    Expected: infinity
    Obtained:1e20
    Remarks: discontinuity occurs at this point for value and derivative and numpy kernel actually evades discontinuity in value but for derivative additional functionality is implemented.
    """
    x = np.pi/2
    X = Variable(x,"X")
    y = np.tan(x)
    Y = Tan(X)
    dy = grad(Y,[X])[0]
    print(dy())
    temp=((1.0/(np.cos(x+1e-12)**2)))
    print(temp)
    assert isinstance(Y,Tan) and y == Y() and (temp-dy()) < 0.000000000001
def test_tan_scalar1():
    """
    Aim:the tan function (value and derivative)
    Expected:0
    Obtained:0
    Remarks:derivatives are not exact to avoid discontinuity
    """
    x = 0
    X = Variable(x,"X")
    y = np.tan(x)
    Y = Tan(X)
    dy = grad(Y,[X])[0]
    print(dy())
    temp=((1.0/(np.cos(x+1e-12)**2)))
    print(temp)
    assert isinstance(Y,Tan) and y == Y() and (temp-dy()) < 0.000000000001
def test_tan_array():
    """
    Aim:the tan function (value and derivative)
    Expected:tan(array)
    Obtained:tan(array)
    Remarks:derivatives are not exact to avoid discontinuity
    """
    x = np.random.rand(4,5)
    y = np.tan(x)
    X = Variable(x,"X")
    Y = Tan(X)    
    dy = grad(Y,[X])[0]
    print(dy())
    temp=((1.0/(np.cos(x)**2)))
    print(temp)
    assert isinstance(Y,Tan) and np.array_equal(Y(),y) and np.all(np.less(np.abs(dy()-temp),0.0000000001))

def test_cosec_array():
    """
    Aim:test the cosec function and derivative
    Expected:cosec(array)
    Obtained:cosec(array+1e-12)
    Remarks:derivatives are not exact to avoid discontinuity
    """
    x = np.random.rand(4,5)
    y = 1.0/np.sin(x+1e-12)
    X = Variable(x,"X")
    Y = Cosec(X)    
    dy = grad(Y,[X])[0]
    print(dy())
    temp=((1.0/(np.sin(x+1e-12)*np.tan(x+1e-12))))
    print(temp)
    assert isinstance(Y,Cosec) and np.array_equal(Y(),y) and np.all(np.less(np.abs(dy()+temp),0.0000000001))


def test_cosec_array_higher():
    """
    Aim:test the cosec function and derivative
    Expected:cosec(array)
    Obtained:cosec(array+1e-12)
    Remarks:derivatives are not exact to avoid discontinuity
    """
    x = np.random.rand(4,5,5)
    y = 1.0/np.sin(x+1e-12)
    X = Variable(x,"X")
    Y = Cosec(X)    
    dy = grad(Y,[X])[0]
    print(dy())
    temp=((1.0/(np.sin(x+1e-12)*np.tan(x+1e-12))))
    print(temp)
    assert isinstance(Y,Cosec) and np.array_equal(Y(),y) and np.all(np.less(np.abs(dy()+temp),0.0000000001))

def test_cosec_scalar():
    """
    Aim:test the cosec function and derivative
    Expected:infinity
    Obtained:1e12
    Remarks:derivatives are not exact to avoid discontinuity
    """
    x = 0
    y = 1.0/(np.sin(x+1e-12))
    X = Variable(x,"X")
    Y = Cosec(X)    
    print(Y())
    print(y)
    dy = grad(Y,[X])[0]
    print(dy())
    temp=((1.0/(np.sin(-1e-12)*np.tan(x+1e-12))))
    print(temp)
    assert isinstance(Y,Cosec) and np.array_equal(Y(),y) and dy() < -1e23 and temp < -1e23


def test_cosec_scalar_1():
    """
    Aim:test the cosec function and derivative
    Expected:1
    Obtained:1
    Remarks:derivatives are not exact to avoid discontinuity
    """
    x = np.pi/2
    y = 1.0/(np.sin(x+1e-12))
    X = Variable(x,"X")
    Y = Cosec(X)    
    print(Y())
    print(y)
    dy = grad(Y,[X])[0]
    print(dy())
    temp=((-1.0/(np.sin(x+1e-12)*np.tan(x+1e-12))))
    print(temp)
    assert isinstance(Y,Cosec) and np.array_equal(Y(),y) and np.all(np.less(np.abs(dy()-temp),0.0000000001))


def test_sec_scalar_1():
    """
    Aim:test the sec function and derivative
    Expected:1
    Obtained:1
    Remarks:difference in derivative constitutes is due to avoiding discontinuities
    """
    x = 0
    y = 1.0/(np.cos(x+1e-12))
    X = Variable(x,"X")
    Y = Sec(X)    
    print(Y())
    print(y)
    dy = grad(Y,[X])[0]
    print(dy())
    temp=np.tan(x+1e-12)/np.cos(x+1e-12)
    print(temp)
    assert isinstance(Y,Sec) and np.array_equal(Y(),y) and np.all(np.less(np.abs(dy()-temp),0.0000000001))

def test_sec_scalar():
    """
    Aim:test the sec function and derivative
    Expected:infinity
    Obtained:1e12
    Remarks:difference in derivative and also value constitutes is due to avoiding discontinuities
    """
    x = np.pi/2
    y = 1.0/(np.cos(x+1e-12))
    X = Variable(x,"X")
    Y = Sec(X)    
    print(Y())
    print(y)
    dy = grad(Y,[X])[0]
    print(dy())
    temp= Sec(X)*Tan(X)
    print(temp())
    assert isinstance(Y,Sec) and np.array_equal(Y(),y) and np.all(np.less(np.abs(dy()-temp()),0.0000000001))


def test_sec_array():
    """
    Aim:test the sec function and derivative
    Expected:sec(array)
    Obtained:sec(array+1e-12)
    Remarks:difference in derivative constitutes is due to avoiding discontinuities
    """
    x = np.random.randn(10,20)
    y = 1.0/(np.cos(x+1e-12))
    X = Variable(x,"X")
    Y = Sec(X)    
    print(Y())
    print(y)
    dy = grad(Y,[X])[0]
    print(dy())
    temp= Sec(X)*Tan(X)
    print(temp())
    assert isinstance(Y,Sec) and np.array_equal(Y(),y) and np.all(np.less(np.abs(dy()-temp()),0.0000000001))



def test_cot_scalar():
    """
    Aim:test the cot function and derivative
    Expected:0
    Obtained:0(almost upto 12 digits)
    Remarks:difference in derivative constitutes is due to avoiding discontinuities
    """
    x = np.pi/2
    y = 1.0/(np.tan(x+1e-12))
    X = Variable(x,"X")
    Y = Cot(X)    
    print(Y())
    print(y)
    dy = grad(Y,[X])[0]
    print(dy())
    temp= -1/((np.sin(x+1e-12)**2))
    print(temp)
    assert isinstance(Y,Cot) and np.array_equal(Y(),y) and np.all(np.less(np.abs(dy()-temp),0.0000000001))


def test_cot_scalar():
    """
    Aim:test the cot function and derivative
    Expected:infinity
    Obtained:1e12
    Remarks:discontinuity occurs at this point for value and derivative and numpy kernel actually evades discontinuity in value but for derivative additional functionality is implemented.
    """
    x = 0
    y = 1.0/(np.tan(x+1e-12))
    X = Variable(x,"X")
    Y = Cot(X)    
    print(Y())
    print(y)
    dy = grad(Y,[X])[0]
    print(dy())
    temp= -1/((np.sin(x+1e-12)**2))
    print(temp)
    assert isinstance(Y,Cot) and np.array_equal(Y(),y) and dy() < -1e23 and temp < -1e23



def test_cot_array():
    """
    Aim: test the cot function and derivative
    Expected: cot (array)
    Obtained: cot(array+1e-12)
    Remarks:difference in derivative constitutes is due to avoiding discontinuities
    """
    x = np.random.randn(22,88)
    y = 1.0/(np.tan(x+1e-12))
    X = Variable(x,"X")
    Y = Cot(X)    
    print(Y())
    print(y)
    dy = grad(Y,[X])[0]
    print(dy())
    temp= -1/((np.sin(x+1e-12)**2))
    print(temp)
    assert isinstance(Y,Cot) and np.array_equal(Y(),y) and np.all(np.less(np.abs(dy()-temp),0.0000001))

#The sigmoid function is a typical case where the derivative is in terms of the function itself. This property is emphasized by the test cases.
def test_sigmoid_scalar():
    """ 
    Aim: Test sigmoid function (value and derivative)
    Expected:0.5
    Obtained:0.5
    Remarks: The values will be exact since the function is itself used as derivative
    """
    x = 0
    y = 1/(1+np.exp(-x))
    X =Variable(x,"X")
    Y = Sigmoid(X)
    dy = grad(Y,[X])[0]
    temp = Sigmoid(X)*(1-Sigmoid(X))
    assert isinstance(Y,Sigmoid) and Y()==y and dy()==temp()

def test_sigmoid_array():
    """
    Aim: test sigmoid function (value and derivative)
    Expected:sigmoid(array)
    Obtained:sigmoid(array)
    Remarks:The values will be exact since the function is itself used as derivative
    """
    x = np.random.rand(3,3,4)
    y = 1/(1+np.exp(-x))
    X =Variable(x,"X")
    Y = Sigmoid(X)
    dy = grad(Y,[X])[0]
    temp = Sigmoid(X)*(1-Sigmoid(X))
    assert isinstance(Y,Sigmoid) and np.array_equal(Y(),y) and np.array_equal(dy(),temp())
#Note for all Inverse trigonometric functions.
#The inverse trigonometric functions work only in specific ranges and there is a possibility that while avoiding infinities the value might be in undefined regions
#This is handled by not using the epsilon in value and derivative also, but the derivative uses functions which already have implemented in a way to avoid infinities. Hence he infinities are handled without any hindrance to teh possible ranges.
#The same is emphasized in the test cases below 
def test_arcsin_scalar():
    """
    Aim:Test the arcsin function (value and derivative)
    Expected: pi/2
    Obtained: pi/2 
    Remarks:discontinuity occurs at derivative but it is evaded by eps.
    """
    x = 1
    y = np.arcsin(x)
    X =Variable(x,"X")
    Y = ArcSin(X)
    dy = grad(Y,[X])[0]
    print(dy())
    x = x 
    temp = 1/(np.sqrt(1-(x*x))+1e-12)
    print(temp)
    assert isinstance(Y,ArcSin) and Y()==y and dy()-temp < 1e-20

def test_arcsin_array():
    """
    Aim:Test the arcsin function (value and derivative)
    Expected:arcsin(array)
    Obtained:arcsin(array)
    Remarks:discontinuity occurs at derivative but it is evaded by eps.
    """
    x = np.random.rand(5,5)
    y = np.arcsin(x)
    X =Variable(x,"X")
    Y = ArcSin(X)
    dy = grad(Y,[X])[0]
    print(dy())
    temp = 1/np.sqrt(1-(x*x))
    assert isinstance(Y,ArcSin) and np.array_equal(Y(),y) and np.all(np.less(np.abs(dy()-temp),0.000000001))

def test_arccos_scalar():
    """
    Aim: Test the arccos function (value and derivative)
    Expected:0
    Obtained:0
    Remarks: typical case where functionality of derivative breaks but the value is more frequent to occur hence it is checked to prevent it. 
    """
    x = 1
    y = np.arccos(x)
    X =Variable(x,"X")
    Y = ArcCos(X)
    dy = grad(Y,[X])[0]
    print(dy())
    x = x 
    temp = -1/(np.sqrt(1-(x*x))+1e-12)
    print(temp)
    assert isinstance(Y,ArcCos) and Y()==y and np.abs(dy()-temp) < 1e-20
def test_arccos_array():
    """
    Aim:test the arccos function (value and derivative)
    Expected: arccos(array)
    Obtained: arccos(array)
    Remarks: To avoid discontinuity of derivative , the values won't exactly match
    """
    x = np.random.rand(5,5)
    y = np.arccos(x)
    X =Variable(x,"X")
    Y = ArcCos(X)
    dy = grad(Y,[X])[0]
    print(dy())
    temp = -1/np.sqrt(1-(x*x))
    assert isinstance(Y,ArcCos) and np.array_equal(Y(),y) and np.all(np.less(np.abs(dy()-temp),0.000000001))

def test_arctan_scalar():
    """
    Aim:test the arc tan function(value and derivative)
    Expected: arctan(1e20)
    Obtained: arctan(1e20)
    Remarks: This case shows that if x tends to infinity , arctan tends to pi/2 and derivative tends to 0, which is expected behaviour
    """
    x = 1e20
    y = np.arctan(x)
    X =Variable(x,"X")
    Y = ArcTan(X)
    dy = grad(Y,[X])[0]
    print(dy())
    x = x 
    temp = 1/(1+(x*x)+1e-12)
    print(temp)
    assert isinstance(Y,ArcTan) and Y()==y and np.abs(dy()-temp) < 1e-20
    

def test_arctan_array():
    """
    Aim: test the arc tan function 
    Expected: arc tan(array)
    Obtained: arc tan(array)
    Remarks: The values and derivatives won't be exact due to avoiding of discontinuity
    """
    x = np.random.randn(5,5,5)
    y = np.arctan(x)
    X =Variable(x,"X")
    Y = ArcTan(X)
    dy = grad(Y,[X])[0]
    print(dy())
    temp = 1/(1+(x*x)+1e-12)
    assert isinstance(Y,ArcTan) and np.array_equal(Y(),y) and np.all(np.less(np.abs(dy()-temp),0.000000000001))

def test_arccot_array():
    """
    Aim: Test the arc cot function (value and derivative)
    Expected: arc cot(array)
    Obtained: arc cot (array + 1e-12 )
    Remarks: The difference arises in derivative due to handling discontinuties
    """
    x = np.random.randn(5,5,5)
    y = np.arctan(1/(x+1e-12))
    X =Variable(x,"X")
    Y = ArcCot(X)
    dy = grad(Y,[X])[0]
    print(dy())
    temp = -1/(1+(x*x)+1e-12)
    assert isinstance(Y,ArcCot) and np.array_equal(Y(),y) and np.all(np.less(np.abs(dy()-temp),0.0000000000001))


def test_arccot_scalar():
    """
    Aim: Test the arc cot function (value and derivative)
    Expected: pi/2 
    Obtained: pi/2
    Remarks: Arc cot of 0 is also nearly calculated close to np.pi/2 , this case is also checked
    """
    x = 0
    y = np.arctan(1/(x+1e-12))
    X =Variable(x,"X")
    Y = ArcCot(X)
    print(Y())
    dy = grad(Y,[X])[0]
    print(dy())
    temp = -1/(1+(x*x)+1e-12)
    assert isinstance(Y,ArcCot) and np.array_equal(Y(),y) and np.all(np.less(np.abs(dy()-temp),0.0000000000001))

def test_arccosec_scalar():
    """
    Aim: Test the arc cosec function (value and derivative)
    Expected: arc cosec(1)
    Obtained: arc cosec (1+1e-12)
    Remarks: This is a place where discontinuity occurs in the derivative and the anomaly is rightly shifted with a small margin of negotiable error.
    """
    x = 1
    y = np.arcsin(1/(x+1e-12))
    X =Variable(x,"X")
    Y = ArcCosec(X)
    #print(Y())
    dy = grad(Y,[X])[0]
    print(dy())
    temp = -1/(x*np.sqrt((x*x)-1)+1e-12)
    print(temp)
    assert isinstance(Y,ArcCosec) and np.array_equal(Y(),y) and np.all(np.less(np.abs(dy()-temp),0.0000000000001))
def test_arccosec_array():
    """
    Aim: Test the Arc cosecant function (derivative and value)
    Expected: arc cosecant value of array
    Obtained: arc cosecant value of array+1e-12
    Remarks:This array covers the whole range of Arccosecant ,i.e 1 to inf(in this case 1e20) ,such that every possible discontinuity is correctly checked
    """
    x = np.random.uniform(1.1,1e20,(5,5))
    y = np.arcsin(1/(x+1e-12))
    X =Variable(x,"X")
    Y = ArcCosec(X)
    #print(Y())
    dy = grad(Y,[X])[0]
    print(dy())
    temp = -1/(x*np.sqrt((x*x)-1)+1e-12)
    print(temp)
    assert isinstance(Y,ArcCosec) and np.array_equal(Y(),y) and np.all(np.less(np.abs(dy()-temp),0.0000000000001))

def test_arcsec_scalar():
    """
    Aim: Test the arcseccant function (value and derivative)
    Expected: arc secant(1)
    Obtained: arc secant(1+1e-12)
    Remarks: This is a typical case where the function breaks infinity in derivative occurs, the discontinuity is evaded by shifting it with a small error.
    """
    x = 1
    y = np.arccos(1/(x+1e-12))
    X =Variable(x,"X")
    Y = ArcSec(X)
    #print(Y())
    dy = grad(Y,[X])[0]
    print(dy())
    temp = 1/(np.abs(x)*np.sqrt((x*x)-1)+1e-12)
    print(temp)
    assert isinstance(Y,ArcSec) and np.array_equal(Y(),y) and np.all(np.less(np.abs(dy()-temp),0.0000000000001))

def test_arcsec_array():
    """
    Aim: Test the arcseccant function (value and derivative)
    Expected: arc secant (array) 
    Obtained: arc secant(array+1e-12 )
    Remarks: This is a case where functionality might break , values taken as 1.1(least possible) and 1e20(system precision inf) , That anomaly avoided by some error
    """
    x = np.random.uniform(1.1,1e20,(3,3))
    y = np.arccos(1/(x+1e-12))
    X =Variable(x,"X")
    Y = ArcSec(X)
    #print(Y())
    dy = grad(Y,[X])[0]
    print(dy())
    temp = 1/(np.abs(x)*np.sqrt((x*x)-1)+1e-12)
    print(temp)
    assert isinstance(Y,ArcSec) and np.array_equal(Y(),y) and np.all(np.less(np.abs(dy()-temp),0.0000000000001))


def test_abs_scalar():
    """
    Aim:To test the absolute function(the value and derivative)
    Expected: 0
    Obtained: 0
    Remarks:The derivatives won't be exact as derivative of abs involves division
    """
    x = 0
    y = np.abs(x)
    X = Variable(x,"X")
    Y = Absolute(X)
    dy = grad(Y,[X])[0]
    temp = 0
    print(np.abs(temp-dy()))
    assert isinstance(Y,Absolute) and np.array_equal(Y(),y) and np.abs(dy()-temp) < 1e-10  

def test_abs_array():
    """
    Aim: To test the absolute function(the value and derivative)
    Expected: absolute value of array
    Obtained: absolute value of array 
    Remarks: The derivatives won't be exact as derivative of abs involves division
    """
    x= np.random.uniform(-1e20,1e20,(5,5))
    y = np.abs(x)
    X = Variable(x,"X")
    Y = Absolute(X)
    dy = grad(Y,[X])[0]
    temp = x / np.abs(x)
    assert isinstance(Y,Absolute) and np.array_equal(Y(),y) and np.all(np.less(np.abs(dy()-temp),0.0000000000001))


