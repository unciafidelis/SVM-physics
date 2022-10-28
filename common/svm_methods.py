import math
import numpy as np
from sklearn.svm import SVC, LinearSVC


class LinearPrecomputed():
    """
    Class that computes by hand the linear kernel
    Returns a numpy array that is the dot product of the input matrices
    """
    def __init__(self, x, y=None):
        self.x = np.array(x)
        if y is None:
            self.y = np.array(x)
        else:
            self.y = np.array(y)

    @staticmethod
    def linear_explicit(v1, v2):
        prod = 0
        for i in range(0, len(v1)):
            prod += v1[i] * v2[i]
        return prod
  
    def explicit_calc(self):
        return np.array([[self.linear_explicit(i, j) for j in self.y ] for i in self.x])

    @staticmethod
    def linear_np(v1, v2):
        return np.dot(v1, v2.T)
   
    def numpy_calc(self):
        return np.array([[self.linear_np(i, j) for j in self.y ] for i in self.x])


class RBFPrecomputed():
    """
    Class that computes by hand the RBF kernel
    Returns a numpy array that is the gaussian transform of the input matrices
    """
    def __init__(self, x, y=None, gamma=0.01):
        self.x = np.array(x)
        self.gamma = gamma
        if y is None:
            self.y = np.array(x)
        else:
            self.y = np.array(y)

    def rbf_explicit(self, v1, v2):
        norm = 0
        for i in range(0, len(v1)):
            norm += (v1[i]-v2[i]) * (v1[i]-v2[i])
        return math.exp(-(norm) * self.gamma)
  
    def explicit_calc(self):
        return np.array([[self.rbf_explicit(i, j) for j in self.y ] for i in self.x])

    def rbf_np(self, v1, v2):
        norm = ((v1-v2).dot(v1-v2))
        return math.exp(-(norm) * self.gamma)
   
    def numpy_calc(self):
        return np.array([[self.rbf_np(i, j) for j in self.y ] for i in self.x])


class PolyPrecomputed():
    """
    Class that computes by hand the polynomial kernel
    Returns a numpy array that is the gaussian transform of the input matrices
    """
    def __init__(self, x, y=None, gamma=1, deg=1, coef=0):
        self.x = np.array(x)
        self.gamma = gamma
        self.deg = deg
        self.coef = coef
        if y is None:
            self.y = np.array(x)
        else:
            self.y = np.array(y)

    def poly_explicit(self, v1, v2):
        prod = 0
        for i in range(0, len(v1)):
            prod += v1[i] * v2[i]
        prod = self.gamma * prod + self.coef
        return math.pow(prod, self.deg)
  
    def explicit_calc(self):
        return np.array([[self.poly_explicit(i, j) for j in self.y ] for i in self.x])

    def poly_np(self, v1, v2):
        prod = np.dot(v1, v2.T)
        prod = self.gamma * prod + self.coef
        return math.pow(prod, self.deg)
   
    def numpy_calc(self):
        return np.array([[self.poly_np(i, j) for j in self.y ] for i in self.x])


class KernelSum():
    """
    Class that computest the linear combination of kernels
    result = a * A + b * B + ...
    Input values are gram matrices (kernel matrices)
    """
    def __init__(self, kernels=[]):
        if len(kernels) == 0:
            print("No input kernels. Goodbye!")
            return None
        else:
            self.kernels = kernels

    def linear_combination(self):
        sum = 0
        for kernel in self.kernels:
            sum += kernel[0] * kernel[1] 
        return sum

class KernelMultiplication():
    """
    Class that computest the product of kernels
    result = (a * A)  * (b * B) * ...
    Input values are gram matrices (kernel matrices)
    """
    def __init__(self, kernels=[]):
        if len(kernels) == 0:
            print("No input kernels. Goodbye!")
            return None
        else:
            self.kernels = kernels

    def product(self):
        # result = 
        # for kernel1 in self.kernels:
        prod1 = self.kernels[0][0] * self.kernels[0][1]
        prod2 = self.kernels[1][0] * self.kernels[1][1]
        print(prod1, type(prod1), prod1.shape)
        # input()

        # 3x3 matrix
        X = [[12,7,3],
            [4 ,5,6],
            [7 ,8,9]]

        # 3x4 matrix
        Y = [[5,8,1,2],
            [6,7,3,0],
            [4,5,9,1]]

        check = np.matmul(X, Y)
        print(check)
        input()    
        result = np.matmul(prod1, prod2)
        # result = [[sum(a*b for a,b in zip(prod1_row,prod2_col)) for prod2_col in zip(*prod2)] for prod1_row in prod1]
            
            # result = [[sum(a*b for a,b in zip(prod1_row,prod1_col)) for prod1_col in zip(*prod1)] for prod1_row in prod1]
        return  result




