import sys
import math
import numpy as np


class LinearPrecomputed():
    """
    Class that computes by hand a linear kernel
    Returns a numpy array that is the dot product of the input matrices
    """
    def __init__(self, x):
        self.x = np.array(x[0])
        if len(x)==1:
            self.y = np.array(x[0])
        elif len(x)==2:
            self.y = np.array(x[1])
        else:
            print("Format not supported. Exiting...")
            sys.exit()            

    @staticmethod
    def linear_explicit(v1, v2):
        prod = 0
        for i in range(0, len(v1)):
            prod += v1[i] * v2[i]
        return prod
  
    def compute(self):
        return np.array([[self.linear_explicit(i, j) for j in self.y ] for i in self.x])

    @staticmethod
    def linear_np(v1, v2):
        return np.dot(v1, v2.T)
   
    def numpy_calc(self):
        return np.array([[self.linear_np(i, j) for j in self.y ] for i in self.x])


class RBFPrecomputed():
    """
    Class that computes by hand a RBF kernel
    Returns a numpy array that is the gaussian transform of the input matrices
    """
    def __init__(self, x, gamma=0.01):
        self.gamma = gamma
        self.x = np.array(x[0])
        if len(x)==1:
            self.y = np.array(x[0])
        elif len(x)==2:
            self.y = np.array(x[1])
        else:
            print("Format not supported. Exiting...")
            sys.exit()

    def rbf_explicit(self, v1, v2):
        norm = 0
        for i in range(0, len(v1)):
            norm += (v1[i] - v2[i]) * (v1[i] - v2[i])
        return math.exp(-(norm) * self.gamma)
  
    def compute(self):
        return np.array([[self.rbf_explicit(i, j) for j in self.y ] for i in self.x])

    def rbf_np(self, v1, v2):
        norm = ((v1 - v2).dot(v1 - v2))
        return math.exp(-(norm) * self.gamma)
   
    def numpy_calc(self):
        return np.array([[self.rbf_np(i, j) for j in self.y ] for i in self.x])


class LaplacePrecomputed():
    """
    Class that computes by hand a Laplace kernel
    Returns a numpy array that is the Laplace transform of the input matrices
    """
    def __init__(self, x, gamma=0.01):
        self.gamma = gamma
        self.x = np.array(x[0])
        if len(x)==1:
            self.y = np.array(x[0])
        elif len(x)==2:
            self.y = np.array(x[1])
        else:
            print("Format not supported. Exiting...")
            sys.exit()

    def rbf_explicit(self, v1, v2):
        norm = 0
        for i in range(0, len(v1)):
            norm += abs((v1[i] - v2[i]))
        return math.exp(-(norm) * self.gamma)
  
    def compute(self):
        return np.array([[self.rbf_explicit(i, j) for j in self.y ] for i in self.x])


class PolyPrecomputed():
    """
    Class that computes by hand a polynomial kernel
    Returns a numpy array that is the gaussian transform of the input matrices
    """
    def __init__(self, x, gamma=1, deg=1, coef=0):
        self.gamma = gamma
        self.deg = deg
        self.coef = coef
        self.x = np.array(x[0])
        if len(x)==1:
            self.y = np.array(x[0])
        elif len(x)==2:
            self.y = np.array(x[1])
        else:
            print("Format not supported. Exiting...")
            sys.exit()

    def poly_explicit(self, v1, v2):
        prod = 0
        for i in range(0, len(v1)):
            prod += v1[i] * v2[i]
        prod = self.gamma * prod + self.coef
        return math.pow(prod, self.deg)
  
    def compute(self):
        return np.array([[self.poly_explicit(i, j) for j in self.y ] for i in self.x])

    def poly_np(self, v1, v2):
        prod = np.dot(v1, v2.T)
        prod = self.gamma * prod + self.coef
        return math.pow(prod, self.deg)
   
    def numpy_calc(self):
        return np.array([[self.poly_np(i, j) for j in self.y ] for i in self.x])


class SigmoidPrecomputed():
    """
    Class that computes by hand a sigmoid kernel
    Returns a numpy array
    """
    def __init__(self, x, gamma=0.01, coef=0):
        self.gamma = gamma
        self.coef = coef
        self.x = np.array(x[0])
        if len(x)==1:
            self.y = np.array(x[0])
        elif len(x)==2:
            self.y = np.array(x[1])
        else:
            print("Format not supported. Exiting...")
            sys.exit()

    def sigmoid_explicit(self, v1, v2):
        prod = 0
        for i in range(0, len(v1)):
            prod += v1[i] * v2[i]
        prod = self.gamma * prod + self.coef
        return math.tanh(prod)
  
    def compute(self):
        return np.array([[self.sigmoid_explicit(i, j) for j in self.y ] for i in self.x])


class KernelSum():
    """
    Class that computest the linear combination of kernels
    result = a * A + b * B + ...
    Input values are gram matrices (kernel matrices)
    """
    def __init__(self, kernels=[]):
        if len(kernels) == 0:
            print("No input kernels. Goodbye!")
            sys.exit()
        else:
            self.kernels = kernels

    def compute(self):
        suma = 0
        for kernel in self.kernels:
            suma += kernel[0] * kernel[1]
        return suma


class KernelProd():
    """
    Class that computes the n-product of kernels
    result = (a*A) * (b*B) * ...
    Input values are gram matrices (kernel matrices)
    """
    def __init__(self, kernels=[]):
        if len(kernels) <= 1:
            print("Not enough kernels. Goodbye!")
            sys.exit()
        else:
            self.kernels = kernels

    def compute(self):
        """
        Computes the Hadamard product of a list of kernels
        matrix product
        """
        prod1 = self.kernels[0][0] * self.kernels[0][1]
        prod2 = self.kernels[1][0] * self.kernels[1][1]
        result = np.multiply(prod1, prod2)
        for i in range(len(self.kernels) - 2):
            temp = self.kernels[i+2][0] * self.kernels[i+2][1]
            result = np.multiply(result, temp)
        return result


def compute_kernel(kernel_fcn, X_train, X_test=None, alphas=[1, 1]):
    """
    Compute the kernel matrix
    """
    kernel_fcn = [*set(kernel_fcn.split("_"))]
    kernel_fcn_temp = kernel_fcn.copy()

    if X_test is None:
        X = [X_train]
    else:
        X = [X_test, X_train]

    matrices_kernels = []
    while len(kernel_fcn_temp) != 0:
        if "rbf" in kernel_fcn_temp:
            kernel_fcn_temp.remove("rbf")
            matrices_kernels.append(RBFPrecomputed(X))

        if "pol" in kernel_fcn_temp:
            kernel_fcn_temp.remove("pol")
            matrices_kernels.append(PolyPrecomputed(X))

        if "sig" in kernel_fcn_temp:
            kernel_fcn_temp.remove("sig")
            matrices_kernels.append(SigmoidPrecomputed(X))

        if "lap" in kernel_fcn_temp:
            kernel_fcn_temp.remove("lap")
            matrices_kernels.append(LaplacePrecomputed(X))

        if "lin" in kernel_fcn_temp:
            kernel_fcn_temp.remove("lin")
            matrices_kernels.append(LinearPrecomputed(X))

        if "sum" in kernel_fcn_temp:
            kernel_fcn_temp.remove("sum")
            
        if  "prd" in kernel_fcn_temp:
            kernel_fcn_temp.remove("prd")
        
    if len(matrices_kernels)==1:
        matrix_kernel = matrices_kernels[0].compute()
    elif len(matrices_kernels)==2:
        kernels = []
        kernels.append((1, matrices_kernels[0].compute()))
        kernels.append((1, matrices_kernels[1].compute()))
        if "sum" in kernel_fcn:
            matrix_kernel = KernelSum(kernels).compute()
        elif "prd" in kernel_fcn:
            matrix_kernel = KernelProd(kernels).compute()
        else:
            print("Kernel not supported. Bye!")
            sys.exit()
    else:
        print("Kernel not supported. Bye!")
        sys.exit()

    return matrix_kernel
